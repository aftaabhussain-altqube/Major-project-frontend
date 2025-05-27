# from CattleScanner.Cattle_inference import CattleInference
# import pandas as pd
# import joblib
#
#
# def predict(side_img, rear_img):
#     path = os.path.dirname(CattleScanner.__file__)
#     cattle_inference = CattleInference(path)
#
#     kpt = cattle_inference.infer_keypoints(side_img, rear_img)
#     side_kpt = kpt[0][0].tolist()
#     rear_kpt = kpt[1][0].tolist()
#     args = cattle_inference.return_args(side_kpt, rear_kpt)
#     pixels = cattle_inference.return_pixels(side_img, cattle_inference.side_segmentation_model)
#     args = args + pixels
#
#     features = ["side_length_shoulderbone", "side_f_girth", "side_r_girth", "rear_width", "cow_pixels",
#                 "sticker_pixels"]
#     target = "weight"
#     model = joblib.load('linear.pkl')
#     new_data = {}
#     for index, i in enumerate(features):
#         new_data[i] = [args[index]]
#     new_df = pd.DataFrame(new_data)
#
#     predicted_weight = model.predict(new_df)[0]
#
#     return predicted_weight


