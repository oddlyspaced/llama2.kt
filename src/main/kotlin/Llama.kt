import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.util.Arrays
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

fun softmax(x: FloatArray, size: Int): FloatArray {
    val maxVal = x.max()
    // exponent and sum
    var sum = 0f
    for (i in 1..<size) {
        x[i] = exp(x[i] - maxVal)
        sum += x[i]
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
    var sortedVocab: Array<TokenIndex>? = null
    var maxTokenLength: UInt = (0).toUInt()
    val bytePieces: CharArray = CharArray(512)
}

fun compareTokens(a: TokenIndex, b: TokenIndex): Int {
    return a.str.compareTo(b.str)
}

fun buildTokenizer(t: Tokenizer, tokenizerPath: String, vocabSize: Int) {
    t.sortedVocab = null // lazy init
    for (i in 0..255) {
        t.bytePieces[i * 2] = i.toChar()
        t.bytePieces[i * 2 + 1] = '\u0000'
    }
    // read the file
    try {
        DataInputStream(FileInputStream(tokenizerPath)).use { input ->
            // Read maxTokenLength
            t.maxTokenLength = input.readInt().toUInt()
            // Read vocabScores and vocab
            for (i in 0..<vocabSize) {
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

fun strLookup(str: String, sortedVocab: Array<TokenIndex>, vocabSize: Int): Int {
    return sortedVocab.find {
        it.str == str
    }?.id ?: -1
}

fun encode(t: Tokenizer, text: String, bos: Int, eos: Int, tokens: IntArray, nTokens: Int) {
    if (t.sortedVocab == null) {
        // lazy load and sort vocab
        t.sortedVocab = Array(t.vocabSize) {
            TokenIndex(t.vocab[it]!!, it)
        }
        t.sortedVocab!!.sortWith { t1, t2 -> compareTokens(t1, t2) }
    }

    var strBuffer = StringBuilder()
    var strLen = 0

    var nTokens = 0
    if (bos > 0) {
        tokens[nTokens++] = 1
    }
    // todo: possible point of failure
    if (text[0] != '\u0000') {
        val dummyPrefix = t.sortedVocab!!.find { it.str == " " }
        tokens[(nTokens++)] = dummyPrefix!!.id
    }

    // utf stuff
    for (c in text) {
        // todo : point of failure
        val byte = c.code and 0xFF // Convert character to byte value

        // Check if the byte is a continuation byte (starts with 10 in UTF-8)
        if (byte and 0xC0 != 0x80) {
            // Not a continuation byte, reset buffer
            strBuffer.clear()
            strLen = 0
        }

        // Append the current byte to the buffer
        strBuffer.append(c)
        strLen += 1

        // Check if the next character is a continuation byte and ensure buffer size
        val nextByte = text.getOrNull(text.indexOf(c) + 1)?.toInt()?.and(0xFF) ?: 0
        if ((nextByte and 0xC0 == 0x80) && strLen < 4) {
            continue
        }

        // Look up the string buffer in the vocabulary
        val id = t.sortedVocab!!.find { it.str == strBuffer.toString() }?.id ?: -1

        if (id != -1) {
            // Found in vocab, add it as a token
            tokens[(nTokens++)] = id
        } else {
            // Byte fallback encoding
            for (i in 0 until strLen) {
                tokens[nTokens++] = strBuffer[i].code + 3
            }
        }
        strBuffer.clear()
        strLen = 0 // Reset for the next codepoint
    }

    while (true) {
        var bestScore = -1e10f
        var bestId = 0
        var bestIdx = -1

        for (i in 0..<nTokens - 1) {
            strBuffer.append(t.vocab[tokens[i]])
            strBuffer.append(t.vocab[tokens[i + 1]])
            val id = t.sortedVocab!!.find { it.str == strBuffer.toString() }?.id ?: -1
            if (id != -1 && t.vocabScores[id] > bestScore) {
                bestScore = t.vocabScores[id]
                bestId = id
                bestIdx = i
            }
        }

        if (bestIdx == -1) {
            break
        }
        tokens[bestIdx] = bestId
        for (i in (bestIdx + 1)..<(nTokens - 1)) {
            tokens[i] = tokens[i + 1]
        }
        nTokens--
    }
    if (eos > 0) {
        tokens[nTokens++] = 2
    }
    strBuffer.clear()
}

data class ProbIndex(
    var prob: Float,
    var index: Int
)

data class Sampler(
    var vocabSize: Int,
    val probIndex: Array<ProbIndex>,
    var temperature: Float,
    var topP: Float,
    var rngState: ULong,
)

fun sampleArgmax(probabilities: FloatArray, n: Int): Int {
    var maxI = 0
    var maxP = probabilities[0]
    for (i in 1..<n) {
        if (probabilities[i] > maxP) {
            maxI = i
            maxP = probabilities[i]
        }
    }
    return maxI
}

// todo: what is this ?
fun sampleMult(probabilities: FloatArray, n: Int, coin: Float): Int {
    var cdf = 0f
    for (i in 0..<n) {
        cdf += probabilities[i]
        if (coin < cdf) {
            return i
        }
    }
    return n - 1
}

fun compare(a: ProbIndex, b: ProbIndex): Int {
    return if (a.prob > b.prob) -1
    else if (b.prob > a.prob) 1
    else 0
}

fun sampleTopP(probabilities: FloatArray, n: Int, topP: Float, probIndex: Array<ProbIndex>, coin: Float): Int {
    var n0 = 0
    val cutoff = (1f - topP) / (n - 1)
    for (i in 0..<n) {
        if (probabilities[i] >= cutoff) {
            probIndex[n0].apply {
                index = i
                prob = probabilities[i]
            }
            n0++
        }
    }
    var probIndex = probIndex.sliceArray(0..<n0)
    probIndex.sortWith { v1, v2 ->
        compare(v1, v2)
    }

    var cumulativeProb = 0f
    var lastIDX = n0 - 1
    for (i in 0..<n0) {
        cumulativeProb += probIndex[i].prob
        if (cumulativeProb > topP) {
            lastIDX = i
            break
        }
    }

    val r = coin * cumulativeProb
    var cdf = 0f
    for (i in 0..lastIDX) {
        cdf += probIndex[i].prob
        if (r < cdf) {
            return probIndex[i].index
        }
    }
    return probIndex[lastIDX].index
}

fun buildSampler(sampler: Sampler, vocabSize: Int, temperature: Float, topP: Float, rngSeed: ULong) {
    sampler.apply {
        this.vocabSize = vocabSize
        this.temperature = temperature
        this.topP = topP
        rngState = rngSeed
    }
}

fun randomU32(state: ULong): UInt {
    // xorshift rng algorithm
    var state = state
    state = state xor (state shr 12)
    state = state xor (state shl 25)
    state = state xor (state shr 27)
    // Use the multiplier constant and right shift to get a 32-bit unsigned integer
    return ((state * (0x2545F4914F6CDD1L).toULong()) shr 32).toUInt()
}

fun randomF32(state: ULong): Float {
    return ((randomU32(state) shr 8).toFloat() / 16777216.0f)
}

fun sample(sampler: Sampler, logits: FloatArray): Int {
    var logits = logits
    val next: Int
    if (sampler.temperature == 0f) {
        next = sampleArgmax(logits, sampler.vocabSize)
    } else {
        for (q in 0..<sampler.vocabSize) {
            logits[q] /= sampler.temperature
        }
        logits = softmax(logits, sampler.vocabSize)
        // random coin flip
        val coin = randomF32(sampler.rngState)
        next = if (sampler.topP <= 0 || sampler.topP >= 1) {
            sampleMult(logits, sampler.vocabSize, coin)
        } else {
            sampleTopP(logits, sampler.vocabSize, sampler.topP, sampler.probIndex, coin)
        }
    }
    return next
}

fun timeInMs(): Long {
    return System.currentTimeMillis()
}

fun generate(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, prompt: String, steps: Int) {
    val numPromptTokens = 0
    val promptTokens = IntArray(prompt.length + 3)

    encode(tokenizer, prompt, 1, 0, promptTokens, numPromptTokens)


}


class Llama {

}

fun main() {
    println("Hello World")
}