import json
from ultralytics import YOLO
# Path to bimodel.pt
bimodel_path = "weights/bimodel.pt"
# Load the YOLO classification model
bimodel = YOLO(bimodel_path)
# Function to check if the image is a leaf or nonleaf
def check_leaf_image(image_path):
    bimodel_results = bimodel.predict(image_path)
    predictions = bimodel_results[0].probs
    class_label = bimodel_results[0].names[predictions.top1]  # Get the predicted class label
    confidence = predictions.top1conf.item()  # Get the confidence score of the predicted label
    return class_label, confidence
# Main crop disease detection function
def crop_disease_detection(crop, image_path):
    # Check if the image is classified as 'leaf' or 'nonleaf'
    classification_result, confidence = check_leaf_image(image_path)
    if classification_result == 'leaf' and confidence >= 0.5:  # Ensure it's confidently classified as leaf
        # Proceed with disease detection
        model1 = YOLO("weights\\DDweight_othercrops.pt")
        model2 = YOLO("weights\\DDweight_4crops.pt")
        selected_model = model2 if crop.lower() in ['cauliflower', 'cassava', 'pear', 'cashew'] else model1
        results = selected_model.predict(image_path)
        # Extract relevant information from the results
        top_names_values = []
        for result in results:
            probs = result.probs.data
            names = result.names
            top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:6]
            top_names_values.extend([(index, names[index], probs[index]) for index in top_indices])
        # Find the disease that matches the crop name
        matching_diseases = [disease for idx, disease, confidence in top_names_values if crop in disease.split('_')]
        output_data = {"crop_name": crop, "results": []}
        output_json_path = "result.json"
        if matching_diseases:
            primary_disease = matching_diseases[0]
            output_data["results"].append({"Your crop is affected with": primary_disease})
            additional_symptoms = [symptoms for idx, symptoms, confidence in top_names_values if symptoms.startswith(crop) and symptoms != primary_disease]
            if additional_symptoms:
                output_data["results"].append({"And also shows some symptoms of": additional_symptoms})
            else:
                output_data["results"].append({"additional_symptoms": None})
        else:
            output_json_path = "result.json"
            first_disease_parts = top_names_values[0][1].split('_')
            second_part = ' '.join(first_disease_parts[1:])
            output_data["results"] = f"Unable to detect the disease properly, but it seems like '{second_part}'. Please provide a better image or check for the disease details in the list provided below."
    else:
        # If the image is classified as 'nonleaf' or low confidence, output an error message
        output_data = {"error": "Please upload a clear leaf image only"}
        output_json_path = "result.json"
    # Save the output to JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    return output_json_path
# Example usage:
crop_name = input("Enter crop name from the dropdown: ")
# image_path = "C:\\Users\\Admin\\Desktop\\Screenshot 2024-11-06 135835.jpg"
image_path=r"C:\Users\ADMIN\Desktop\test data\leaff.jpg"

output_json_file = crop_disease_detection(crop_name, image_path)
print(f"Results saved to: {output_json_file}")