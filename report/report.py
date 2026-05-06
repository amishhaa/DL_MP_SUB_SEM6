import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

y_test = np.load("../report/y_test.npy")
y_pred_lstm = np.load("../report/y_pred_lstm.npy")
y_pred_bilstm = np.load("../report/y_pred_bilstm.npy")

report_lstm = classification_report(y_test, y_pred_lstm, output_dict=True)
report_bilstm = classification_report(y_test, y_pred_bilstm, output_dict=True)

cm_lstm = confusion_matrix(y_test, y_pred_lstm)
cm_bilstm = confusion_matrix(y_test, y_pred_bilstm)

def build_table(report):
    rows = ""
    for label in ['0', '1']:
        r = report[label]
        rows += f"""
        <tr>
            <td>{label}</td>
            <td>{r['precision']:.2f}</td>
            <td>{r['recall']:.2f}</td>
            <td>{r['f1-score']:.2f}</td>
        </tr>
        """
    return rows

html = f"""
<html>
<head>
    <title>Log Anomaly Detection Report</title>
    <style>
        body {{
            font-family: Arial;
            margin: 40px;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 60%;
            margin-bottom: 30px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 10px;
            text-align: center;
        }}
        th {{
            background-color: #f4f4f4;
        }}
    </style>
</head>

<body>

<h1>Log Anomaly Detection Report</h1>

<h2>LSTM Results</h2>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{build_table(report_lstm)}
</table>

<p><b>Confusion Matrix:</b> {cm_lstm.tolist()}</p>

<h2>BiLSTM Results</h2>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{build_table(report_bilstm)}
</table>

<p><b>Confusion Matrix:</b> {cm_bilstm.tolist()}</p>

</body>
</html>
"""

os.makedirs("../report", exist_ok=True)

with open("../report/report.html", "w") as f:
    f.write(html)

print("Report saved at report/report.html")