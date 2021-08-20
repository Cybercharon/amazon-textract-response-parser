import cv2
import json
data = open('../tests/pii_test/pii_image_example-png-response.json', 'r')
textract_response = json.load(data)

image = cv2.imread('../tests/pii_test/pii_image_example.png')
height, width, channels = image.shape
bbox = textract_response[0].get('Blocks')[8].get('Geometry').get('BoundingBox')

start_point = (int(bbox['Top'] * height), int(bbox['Left'] * width))
end_point = (int(bbox['Width'] * height), int(bbox['Height'] * width))
color = (0,0,0)
thickness = -1
# Create a window
cv2.namedWindow('image')

wait_time = 33

output = cv2.rectangle(image, start_point, end_point, color, thickness)
while True:
    # Display output image
    cv2.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()