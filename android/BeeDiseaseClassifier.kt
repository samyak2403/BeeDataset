package com.beeguard.hivesense.ml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * BeeGuard HiveSense - On-Device Honeybee Disease Classifier
 * 
 * Powered by MobileNetV2 Transfer Learning 7-Class Model exported to TensorFlow Lite.
 * Optimized for real-time edge execution on mobile devices and smart hive cameras.
 *
 * Supported Classes:
 * 1. Varroa Mites
 * 2. American Foulbrood
 * 3. European Foulbrood
 * 4. Chalkbrood
 * 5. Sacbrood
 * 6. Nosema
 * 7. Healthy
 */
class BeeDiseaseClassifier(
    private val context: Context,
    private val modelFileName: String = "bee_disease_model_quantized.tflite",
    private val labelFileName: String = "labels.txt"
) : AutoCloseable {

    companion object {
        const val INPUT_SIZE = 224
        const val PIXEL_CHANNELS = 3 // RGB
        const val BYTES_PER_FLOAT = 4
    }

    private var interpreter: Interpreter? = null
    private val labels: List<String> = loadLabels(context.assets, labelFileName)
    private val numClasses: Int = labels.size

    // Preallocated native direct byte buffers for maximum mobile throughput
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(
        1 * INPUT_SIZE * INPUT_SIZE * PIXEL_CHANNELS * BYTES_PER_FLOAT
    ).apply {
        order(ByteOrder.nativeOrder())
    }

    private val outputBuffer: Array<FloatArray> = Array(1) { FloatArray(numClasses) }

    init {
        val modelByteBuffer = loadModelFile(context.assets, modelFileName)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Enable NNAPI or GPU Delegate when hardware acceleration is available
            useNNAPI = false 
        }
        interpreter = Interpreter(modelByteBuffer, options)
    }

    /**
     * Diagnostic classification result container.
     */
    data class Recognition(
        val label: String,
        val confidence: Float,
        val classIndex: Int
    )

    /**
     * Executes inference on a given camera frame or photo Bitmap.
     *
     * @param bitmap Raw input image from camera or gallery.
     * @return Ranked list of disease probabilities sorted by confidence descending.
     */
    fun classifyImage(bitmap: Bitmap): List<Recognition> {
        val activeInterpreter = interpreter
            ?: throw IllegalStateException("BeeDiseaseClassifier has been closed or uninitialized.")

        // 1. Resize and crop bitmap to 224x224
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // 2. Load and preprocess pixels into direct ByteBuffer [0.0f, 255.0f]
        // Note: The Keras model includes the MobileNetV2 preprocess_input layer internally.
        inputBuffer.rewind()
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        var pixelIdx = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixelValue = intValues[pixelIdx++]
                // Extract RGB components as Float [0, 255]
                val r = ((pixelValue shr 16) and 0xFF).toFloat()
                val g = ((pixelValue shr 8) and 0xFF).toFloat()
                val b = (pixelValue and 0xFF).toFloat()

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }

        // 3. Run TFLite inference
        activeInterpreter.run(inputBuffer, outputBuffer)

        // 4. Map output probabilities to labels
        val probabilities = outputBuffer[0]
        val results = mutableListOf<Recognition>()
        for (i in 0 until numClasses) {
            val label = if (i < labels.size) labels[i] else "Class $i"
            results.add(Recognition(label = label, confidence = probabilities[i], classIndex = i))
        }

        // Sort descending by diagnostic confidence
        return results.sortedByDescending { it.confidence }
    }

    private fun loadModelFile(assetManager: AssetManager, fileName: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(assetManager: AssetManager, fileName: String): List<String> {
        return assetManager.open(fileName).bufferedReader(Charsets.UTF_8).readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
    }
}

