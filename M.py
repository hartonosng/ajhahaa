def calculate_accuracy_with_keyword(ground_truths, predictions):
    """
    Menghitung accuracy dengan memastikan jika kata kunci dari ground truth ada di prediksi,
    maka akurasi dianggap 100%.
    
    :param ground_truths: List of ground truth strings
    :param predictions: List of prediction strings
    :return: Float accuracy score (0-100)
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("Jumlah ground truth dan prediksi harus sama.")
    
    total_score = 0
    for gt, pred in zip(ground_truths, predictions):
        # Jika kata kunci ground truth ada di prediksi, akurasi dianggap 100%
        if gt in pred:
            total_score += 1  # 100% untuk pasangan ini
        else:
            # Jika tidak, gunakan similarity ratio
            similarity = SequenceMatcher(None, gt, pred).ratio()
            total_score += similarity  # Skor parsial (0-1)
    
    # Rata-rata skor dikonversi ke persentase
    accuracy = (total_score / len(ground_truths)) * 100
    return accuracy

# Contoh penggunaan
ground_truths = ["Jakarta"]
predictions = ["Kompleks detail di Jakarta"]

accuracy = calculate_accuracy_with_keyword(ground_truths, predictions)
print(f"Accuracy: {accuracy:.2f}%")
