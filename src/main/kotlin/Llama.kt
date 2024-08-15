import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import kotlin.math.exp
import kotlin.math.sqrt

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
    var x: Float,
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

data class Transformer(
    val config: Config,
    val weights: TransformerWeights,
    val state: RunState,
)

fun rmsnorm(x: FloatArray, weights: FloatArray): FloatArray {
    val output = FloatArray(x.size)
    var ss = 0.0f
    for (i in weights.indices) {
        ss += x[i] * x[i]
    }
    ss /= weights.size
    ss += 1e-5f
    ss = 1f / sqrt(ss)
    // normalize and scale
    for (i in weights.indices) {
        output[i] = weights[i] * (ss * x[i])
    }
    return output
}

fun softmax(x: FloatArray): FloatArray {
    val maxVal = x.max()
    // exponent and sum
    var sum = 0f
    x.forEachIndexed { index, value ->
        x[index] = exp(value - maxVal)
        sum += x[index]
    }
    // normalize
    for (i in x.indices) {
        x[i] /= sum
    }
    return x
}

fun matmul(x: FloatArray, w: FloatArray, n: Int, d: Int): FloatArray {
    // W (d,n) @ x (n,) -> xout (d,)
    val output = FloatArray(d)
    for (i in 0..<d) {
        var v = 0f
        for (j in 0..<n) {
            v += w[i * n + j] * x[j]
        }
        output[i] = v
    }
    return output
}

//fun forward(transformer: Transformer, token: Int, pos: Int) {
//    val config = transformer.config // p
//    val weights = transformer.weights // w
//    val state = transformer.state // s
//    val x = state.x
//    val dim = config.dim
//    val kvDim = (config.dim * config.kvHeadCount) / config.headCount
//    val kvMul = config.headCount / config.kvHeadCount
//    val hiddenDim = config.hiddenDim
//    val headSize = dim / config.headCount
//
//    // copy the token embedding into x
//    val contentRow = weights.tokenEmbeddingTable[token * dim]
//    state.x = contentRow
//
//    for (l in 0..<config.layerCount) {
//         rmsnorm(x, )
//    }
//}

// Byte Pair Encoding

data class TokenIndex(
    val str: String,
    val id: Int,
)

class Tokenizer(val vocabSize: Int) {
    val vocab = Array<String?>(vocabSize) { null }
    val vocabScores = FloatArray(vocabSize)
    var sortedVocab: TokenIndex? = null
    var maxTokenLength: Int = 0
    val bytePieces: CharArray = CharArray(512)
}

fun compareTokens(a: TokenIndex, b: TokenIndex): Int {
    return a.str.compareTo(b.str)
}

fun buildTokenizer(t: Tokenizer, tokenizerPath: String) {
    t.sortedVocab = null // lazy init
    for (i in 0..255) {
        t.bytePieces[i * 2] = i.toChar()
        t.bytePieces[i * 2 + 1] = '\u0000'
    }
    // read the file
    try {
        DataInputStream(FileInputStream(tokenizerPath)).use { input ->
            // Read maxTokenLength
            t.maxTokenLength = input.readInt()
            // Read vocabScores and vocab
            for (i in 0..<t.vocabSize) {
                t.vocabScores[i] = input.readFloat()
                // Read the length of the next string
                val len = input.readInt()
                // Read the string and assign to vocab[i]
                val byteArray = ByteArray(len)
                input.readFully(byteArray)
                t.vocab[i] = String(byteArray)
            }
        }
    } catch (e: IOException) {
        System.err.println("Failed to load tokenizer: ${e.message}")
    }
}

fun decode(t: Tokenizer, prevToken: Int, token: Int): String? {
    var piece = t.vocab[token]
    if (prevToken == 1) {
        piece = piece?.trimStart()
    }
    // todo: implement hex parsing
    return piece
}

class Llama {

}

fun main() {
    println("Hello World")
}