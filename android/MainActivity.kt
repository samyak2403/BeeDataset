package com.beeguard.hivesense

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.beeguard.hivesense.ml.BeeDiseaseClassifier
import java.io.InputStream

/**
 * Example Android Activity demonstrating real-time disease classification
 * using BeeDiseaseClassifier.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var classifier: BeeDiseaseClassifier
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var diagnoseButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.sampleBeeImage)
        resultTextView = findViewById(R.id.diagnosisResultText)
        diagnoseButton = findViewById(R.id.btnRunDiagnosis)

        // Initialize Classifier with exported TFLite model and labels.txt
        try {
            classifier = BeeDiseaseClassifier(
                context = this,
                modelFileName = "bee_disease_model_quantized.tflite",
                labelFileName = "labels.txt"
            )
            resultTextView.text = "BeeGuard HiveSense Classifier Ready."
        } catch (e: Exception) {
            Log.e("BeeGuard", "Error initializing TFLite classifier", e)
            resultTextView.text = "Initialization error: ${e.localizedMessage}"
        }

        diagnoseButton.setOnClickListener {
            runSampleDiagnosis()
        }
    }

    private fun runSampleDiagnosis() {
        try {
            // Load a sample test image from assets
            val inputStream: InputStream = assets.open("test_sample.png")
            val sampleBitmap: Bitmap = BitmapFactory.decodeStream(inputStream)
            imageView.setImageBitmap(sampleBitmap)

            // Perform Edge AI Inference
            val startTime = System.currentTimeMillis()
            val predictions = classifier.classifyImage(sampleBitmap)
            val latencyMs = System.currentTimeMillis() - startTime

            // Surface top diagnostic result
            val topPrediction = predictions.firstOrNull()
            if (topPrediction != null) {
                val report = StringBuilder()
                    .append("=== BeeGuard Diagnostic Result ===\n")
                    .append("Primary Diagnosis: ${topPrediction.label}\n")
                    .append("Confidence: ${(topPrediction.confidence * 100).format(2)}%\n")
                    .append("Inference Latency: ${latencyMs} ms\n\n")
                    .append("Top Predictions:\n")

                predictions.take(3).forEach { rec ->
                    report.append("- ${rec.label}: ${(rec.confidence * 100).format(2)}%\n")
                }
                resultTextView.text = report.toString()
            }
        } catch (e: Exception) {
            Log.e("BeeGuard", "Diagnosis execution failed", e)
            resultTextView.text = "Inference failed: ${e.localizedMessage}"
        }
    }

    private fun Float.format(digits: Int) = "%.${digits}f".format(this)

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}

