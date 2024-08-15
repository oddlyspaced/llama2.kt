data class Config(
    val dim: Int, // transformer dimension
    val hiddenDim: Int, // for feed forward network layers
    val layerCount: Int, // layer count
    val headCount: Int, // query head count
    val kvHeadCount: Int, // key / value head count (can be < query because of multiquery)
    val vocabSize: Int, // vocabulary size, usually 256 (byte level)
    val seqLen: Int, // max sequence length
)

data class TransformerWeights(
    val tokenEmbeddingTable: FloatArray,
    // weights for rmsnorms
    val rmsAttWeight: FloatArray, // rmsnorm weights
    val rmsFFNWeight: FloatArray,
    val wq: FloatArray,
    val wk: FloatArray,
    val wv: FloatArray,
    val wo: FloatArray,
    val w1: FloatArray,
    val w2: FloatArray,
    val w3: FloatArray,
    val rmsFinalWeight: FloatArray,
    val wcls: FloatArray,
)

data class RunState(
    val x: Float,
    val xb: Float,
    val xb2: Float,
    val hb: Float,
    val hb2: Float,
    val q: Float,
    val k: Float,
    val v: Float,
    val att: Float,
    val logits: Float,
    val keyCache: FloatArray,
    val valueCache: FloatArray,
)

class Llama {

}

fun main() {
    println("Hello World")
}