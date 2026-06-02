from api.v1.predict.predict_service import predict_flood


def predict_controller(input_data: list[dict]):
    return predict_flood(input_data)