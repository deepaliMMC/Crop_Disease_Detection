#pip install ultralytics (in terminal)
import json
from ultralytics import YOLO

def crop_disease_detection(crop, image_path):
    model1 = YOLO("C:\\Users\\Admin\\Downloads\\crop disease detection model weights\\DDweight_othercrops.pt")
    model2 = YOLO("C:\\Users\\Admin\\Downloads\\crop disease detection model weights\\DDweight_4crops.pt")

    if crop.lower() in ['cauliflower', 'cassava', 'pear', 'cashew']:
        selected_model = model2
    else:
        selected_model = model1

    results = selected_model.predict(image_path)

    # Extract relevant information from the results
    top_names_values = []
    for result in results:
        probs = result.probs.data
        # print(f'results probs data is :{result}')
        names = result.names
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:6]
        top_names_values.extend([(index, names[index], probs[index]) for index in top_indices])

    # Find the disease that matches the crop name
    matching_diseases = [disease for idx, disease, confidence in top_names_values if crop in disease.split('_')]

    output_data = {"crop_name": crop, "results": []}

    if matching_diseases:
        output_json_path = "C:\\Users\\Admin\\Downloads\\crop disease detection model weights\\result.json"
        primary_disease = matching_diseases[0]
        output_data["results"].append({"Tour crop is affected by": primary_disease})

        additional_symptoms = [symptoms for idx, symptoms, confidence in top_names_values if symptoms.startswith(crop) and symptoms != primary_disease]

        if additional_symptoms:
            output_data["results"].append({"And also shows some symptoms of": additional_symptoms})
        else:
            output_data["results"].append({"additional_symptoms": None})
    else:
        output_json_path = "C:\\Users\\Admin\\Downloads\\crop disease detection model weights\\error.json"
        first_disease_parts = top_names_values[0][1].split('_')
        second_part = ' '.join(first_disease_parts[1:])
        output_data = {"crop_name": crop, "results": ''}
        output_data["results"] = f"Unable to detect the disease properly, but it seems like '{second_part}'. Please provide a better image or check for the disease details in the list provided below."

    with open(output_json_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    return output_json_path

# Example usage:
# crop_name = "tom"
crop_name = input("Enter crop name from the dropdown: ")
# image_path = input("Enter image path: ")
image_path= "C:\\Users\\Admin\\Desktop\\DISEASE DETECT KAR\\tomato_2.JPG"
output_json_file = crop_disease_detection(crop_name, image_path)

print(f"Results saved to: {output_json_file}")


