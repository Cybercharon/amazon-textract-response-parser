import boto3
import cv2
import json


class PII:
    # Redact always over rides flag
    # Initialize the class setting the defaults.
    # This needs to pass in the textract json
    def __init__(self,
                 # Language for comprehend to use
                 language: str = 'en',
                 # The score you want to accept from comprehend
                 confidence_score: float = 0.01,
                 # The PII types you want to filter out if left blank it will flag all types
                 types: list = None,
                 # Flag to return just T/F
                 flag_pii: bool = False,
                 # Flag to redact the PII in the image
                 return_redacted_image: bool = True
                 ):
        self.comprehend = boto3.client('comprehend')
        self.img = None
        self.redact_img = None
        self.height = None
        self.width = None
        self.lang_str = language
        self.keyList = []
        self.offsetlist = []
        self.textract_response = None
        self.comprehend_response = None
        self.filtered_comprehend = None
        self.text_block = ""
        self.confidence = confidence_score
        self.type_filter = types
        self.flag = flag_pii
        self.return_image = return_redacted_image

    def __build_bbox_start_end(self, bbox):
        x1 = bbox['Left'] * self.width
        y1 = bbox['Top'] * self.height-2
        x2 = x1 + (bbox['Width']*self.width)+5
        y2 = y1 + (bbox['Height']*self.height)+2
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))

        return start_point, end_point

    def __build_blackbox(self, start, end):
        color = (0,0,0)
        thickness = -1
        self.redact_img = cv2.rectangle(self.img, start, end, color, thickness)

    def test_render(self):
        cv2.namedWindow('image')
        wait_time = 33
        while True:
            # Display output image
            cv2.imshow('image', self.redact_img)

            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # This takes the filtered comprehend response
    # makes blackbox on images for the PII
    # TODO build more test cases to ensure the logic here works.
    def __find_pii_from_filtered_comprehend(self):
        for detection in self.filtered_comprehend:
            for i in range(0, len(self.offsetlist)-1):
                if detection.get('BeginOffset') <= self.offsetlist[i] <= detection.get('EndOffset'):
                    self.__build_blackbox(self.keyList[i].get('start_point'), self.keyList[i].get('end_point'))
                    print(f"{self.keyList[i].get('word')}, {self.keyList[i].get('start_point')}, {self.keyList[i].get('end_point')}")
        return

    # This function accepts the comprehend response
    # Creates a filtered response based on score and type
    def __filter_pii(self):
        self.filtered_comprehend = []
        for entity in self.comprehend_response.get('Entities'):
            if (self.confidence <= entity.get('Score') and self.type_filter is None) \
                    or (self.confidence <= entity.get('Score') and entity.get('Type') in self.type_filter):
                self.filtered_comprehend.append(entity)
        return self.filtered_comprehend

    # Reconstruct the text into a multi line paragraph
    # Builds keyDict
    def __reconstruct_doc(self):
        total_length = 0
        for blocks in self.textract_response:
            for block in blocks.get('Blocks'):
                if block.get('BlockType') == 'WORD':
                    word = block.get('Text')
                    self.offsetlist.append(total_length)
                    self.text_block = f"{self.text_block}{word} "
                    start, end = self.__build_bbox_start_end(block.get('Geometry').get('BoundingBox'))
                    self.keyList.append({
                        'word': word,
                        'length': len(word),
                        'BeginOffset': total_length,
                        'start_point': start,
                        'end_point': end
                    })
                    total_length += len(word) + 1
            return self.text_block

    # This function accepts the json from textract and passes it to __reconstruct_doc to
    # generate a multi line string then passes that to comprehend and returns the response
    def __get_comprehend(self):
        if len(self.text_block) == 0:
            self.__reconstruct_doc()
        self.comprehend_response = self.comprehend.detect_pii_entities(
            Text=self.text_block,
            LanguageCode=self.lang_str
        )

        return self.comprehend_response

    def ExecuteTextract2Comprehend(self, image):
        self.img = cv2.imread(image)
        self.height, self.width, _ = self.img.shape

        ####### SHORT CIRCUT TEXTRACT FOR NOW
        data = open('../tests/pii_test/pii_image_example-png-response.json', 'r')
        textract_response = json.load(data)
        ####### SHORT CIRCUT TEXTRACT FOR NOW
        self.textract_response = textract_response
        self.__get_comprehend()
        self.__filter_pii()
        self.__find_pii_from_filtered_comprehend()

        return self.redact_img


if __name__ == '__main__':
    textract2comprehend = PII(language='en')
    pii_response = textract2comprehend.ExecuteTextract2Comprehend('../tests/pii_test/pii_image_example.png')

    print(pii_response)
