const tf = require('@tensorflow/tfjs-node-gpu')
// const tfvis = require('@tensorflow/tfjs-vis');

const mobileNetPath = 'http://localhost:3005/train/model.json'

const main = async () => {
  try {
    const mobileNet = await tf.loadGraphModel(mobileNetPath)
    mobileNet.summary()
  } catch (e) {
    console.log('--------', e)
  }
}
module.exports = main

