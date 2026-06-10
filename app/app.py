import json
from flask import Flask, render_template, request, jsonify
from predictor import predict, build_row, get_risk_tier
from explainer import explain
from segmentation import get_segment
from shap_utils import get_shap_explanation, make_waterfall_chart

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    form_data = request.form.to_dict()
    result    = predict(form_data)

    explanation  = explain(result["_raw"], result["risk"])
    segment      = get_segment(result["_raw"], result["probability"])
    shap_factors = get_shap_explanation(result["df_row_scaled"], top_n=5)
    shap_chart   = make_waterfall_chart(result["df_row_scaled"])

    return render_template(
        "result.html",
        probability    = result["probability"],
        risk           = result["risk"],
        risk_color     = result["risk_color"],
        reasons        = explanation["reasons"],
        suggestions    = explanation["suggestions"],
        segment        = segment,
        shap_factors   = shap_factors,
        shap_chart     = shap_chart,
        form_data      = form_data,
        form_data_json = json.dumps(form_data),
    )


@app.route("/whatif", methods=["POST"])
def whatif():
    data          = request.get_json()
    original_prob = float(data.get("original_prob", 0))
    modified_form = data.get("form_data", {})

    result       = predict(modified_form)
    new_prob     = result["probability"]
    delta        = round(new_prob - original_prob, 1)
    new_risk, new_risk_color = get_risk_tier(new_prob / 100)
    shap_factors = get_shap_explanation(result["df_row_scaled"], top_n=5)

    return jsonify({
        "new_prob":       new_prob,
        "delta":          delta,
        "new_risk":       new_risk,
        "new_risk_color": new_risk_color,
        "shap_factors":   shap_factors,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)