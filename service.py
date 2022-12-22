import bentoml

import numpy as np
from bentoml.io import Image
from bentoml.io import JSON

runner = bentoml.keras.get("keras_xception_final:latest").to_runner()

svc = bentoml.Service("keras_xception_final", runners=[runner])

@svc.api(input=Image(), output=JSON())
async def predict(img):

    from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

    classes = np.asarray(['cup', 'fork', 'glass', 'knife', 'plate', 'spoon'])
    img = img.resize((299, 299))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = await runner.async_run(arr)
    #print(decode_predictions(preds, top=6))
    #return decode_predictions(preds, top=6)[0]
    result = classes[preds.argmax(axis=1)]
    return result