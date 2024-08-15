data class Config(
    val dim: Int, // transformer dimension
    val hiddenDim: Int, // for feed forward network layers
    val layerCount: Int, // layer count
    val headCount: Int, // query head count
    val kvHeadCount: Int, // key / value head count (can be < query because of multiquery)
    val vocabSize: Int, // vocabulary size, usually 256 (byte level)
    val seqLen: Int, // max sequence length
)

class Llama {

}

fun main() {
    println("Hello World")
}