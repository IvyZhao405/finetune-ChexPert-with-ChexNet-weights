from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB3 as Net
from src import input_fn,config as cfg
import traceback
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This API listens to incoming request for the approach, loads the appropriate model based on the approach sent in the request, predicts based on the inference files and creates a CSV with predictions
# in resource folder.
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if True:

        try:
            json_request = request.json
            batch_size = cfg.chexper_params['batch_size']
            infer = cfg.input_file['infer']
            approach = json_request.get("approach")
            model_name = cfg.output_path[approach]['model_name']
            model_path = os.path.join('output', model_name)
            model_dir = cfg.output_path[approach]['directory']

            test_x = input_fn.process_inference_data(file_path=infer)
            test_input = input_fn.input_fn_multi_output(False, test_x, None, batch_size)
            test_steps = len(test_x) // batch_size
            model = load_model(model_path, compile=False)
            result = model.predict(test_input, steps=test_steps)
            columns = ["Pred_Atelectasis", "Pred_Cardiomegaly", "Pred_Consolidation", "Pred_Edema",
                       "Pred_PleuralEffusion"]
            save_name = 'resources/' + 'inference_results_' + 'weighted_' + '5_' + model_dir + '.csv'

            y_pred = []
            for i in range(5):
                y_pred.append((model.predict(test_input)[i] > 0.5).astype(int))

            result_hard_labels_df = pd.DataFrame()
            for i in range(len(result)):
                result_hard_labels_df[columns[i]] = y_pred[i].flatten()
            result_hard_labels_df['image_path'] = test_x[:len(result_hard_labels_df)]

            result_hard_labels_df.to_csv(save_name)
            return jsonify({'inference': 'completed successfully'})


        except:
            print(json_request)
            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':

    app.run(debug=True)