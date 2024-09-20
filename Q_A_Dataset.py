from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import cv2
from array import array
import os
from PIL import Image
import sys
import time
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
import numpy as np 
from io import BytesIO
from msrest.authentication import CognitiveServicesCredentials
import pytesseract
tesseract_path = os.path.join("# replace the directory with your path where pytesseract is installed.") 
pytesseract.pytesseract.tesseract_cmd = tesseract_path
import json


class Asset_Manager_Q_A():

    
    
    
    
    
    
    
    def answer_extraction(image_path):
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        end_coordinate=[]
        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilate = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            end_coordinate.append(y)
        
        height, width, _ = image.shape
        end_coordinate.append(height)
        return end_coordinate
    
    def check_mark(image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Convert to HSV (Hue, Saturation, Value) color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a red rectangle around each detected green tick mark
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h+10), (0, 0, 255), 2)

        
        return(y,y+h+10)
    

    def extract_image_section(image_path, y_start, y_end):
        # Read the image
        image = cv2.imread(image_path)
    
        # Get the height of the image
        height, width, _ = image.shape
    
        # Ensure y_start and y_end are within bounds
        y_start = max(0, min(height-1, y_start))
        y_end = max(0, min(height-1, y_end))
    
        # Exclude the line at the start
        if y_start < height - 1:
            y_start += 1
    
        # Extract the region defined by the y-axis range
        extracted_region = image[y_start-5:y_end,:]
    
        return extracted_region
    
    def perform_ocr(image):
        recognized_text = pytesseract.image_to_string(image, lang='DEU')
        return recognized_text


    def generate_image_data(question,answer):
        

            data={
                    
                    "Question": question ,
                    "Answer": answer.replace("\n", "" ),
                    
    
            }

            return data

subscription_key = "your subscription key"
endpoint = "your Azure End point"

# Authenticate the client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

image_directory='your image directory'
for filename in os.listdir(image_directory):
    
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        image_file_path = os.path.join(image_directory, filename)
        


        end_coordinate=(np.sort(Asset_Manager_Q_A.answer_extraction(image_file_path)))
        
        start,filter=Asset_Manager_Q_A.check_mark(image_file_path)
        

        for i in end_coordinate:
            if i>filter:
                end_val=i
                break

        extracted_section=Asset_Manager_Q_A.extract_image_section(image_file_path,start,end_val)
        
    
        with open(image_file_path, "rb") as image_stream:
        # Use the Computer Vision client to extract text
            read_response = computervision_client.read_in_stream(image_stream, raw=True)

        # Get the operation location (URL with ID as last appendage)
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Retrieve the results (this may take some time)
        while True:
            result = computervision_client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
             break
            time.sleep(1)

        # Extract the text lines
        text_lines = []
        if result.status == OperationStatusCodes.succeeded:
            for read_result in result.analyze_result.read_results:
                for line in read_result.lines:
                    text_lines.append(line.text)

        # Define a set of keywords or patterns that indicate a question
        question_indicators = ['?', ')', '...']

    # Initialize an empty list to store potential questions and their previous lines
        questions_with_context = []

        # Track previous lines
        previous_lines = []

        # Search for lines that likely contain a question
        for line in text_lines:
            lower_line = line.lower()
            if any(indicator in lower_line for indicator in question_indicators):
                # Combine previous lines if they don't contain "Thema" or "Siemens"
                context_lines = [prev_line for prev_line in previous_lines if "thema" not in prev_line.lower() and 
                                "test" not in prev_line.lower() and "SIEMENS" not in prev_line 
                                and "Ingenuity" not in prev_line and ":" not in prev_line]
        
                # Find the position of the last occurring indicator in the line
                last_index = max((lower_line.rfind(indicator) for indicator in question_indicators if indicator in lower_line), default=-1)
                if last_index != -1:
                    line = line[:last_index + 1].rstrip()  # Include the indicator itself
        
                context_lines.append(line.strip())
                questions_with_context.append(" ".join(context_lines))
                previous_lines = []  # Reset previous lines after finding a question
            else:
                previous_lines.append(line.strip())

        # Print the first question with context if it exists
        if questions_with_context:
            first_question_with_context = questions_with_context[0]
            print("Question:")
            print(first_question_with_context)
        else:
            print("No questions found.")
    
        answers=Asset_Manager_Q_A.perform_ocr(extracted_section)
        print("Answer:")
        print(answers)
        base_path = "add your JSON folder location." 
        folder_name = "JSON_FOLDER"
        json_directory= os.path.join(base_path, folder_name)
        data=Asset_Manager_Q_A.generate_image_data(first_question_with_context,answers)
        json_filename = os.path.join(json_directory, f"{image_file_path.split('.')[0]}.json") 
        with open(json_filename, 'a',encoding="utf-8") as json_file:
            json.dump(data, json_file,ensure_ascii=False)
            print(f"JSON file '{json_filename}' written successfully.")