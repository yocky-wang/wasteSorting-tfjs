const getData = require('./data')
const test = require('./test')
const testData = require('./testData')
const tf = require('@tensorflow/tfjs-node-gpu')
const fs = require('fs')
const path = require('path')
// const tfvis = require('@tensorflow/tfjs-vis');

const dirPath = './train-垃圾目录'
const outPath = './output-5-6'
const mobileNetPath = 'http://localhost:3004/train/mobile.json'

// const mobileNetPath = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json'

// tfvis.setOrchestrator(tf);

// // 创建一个包含损失和准确率的容器
// const surface = { name: 'Loss and Accuracy', tab: 'Model Training' };
// tfvis.show.fitCallbacks(surface, ['loss', 'acc'], { callbacks: ['onEpochEnd'] });

// 定义一个用于绘制损失和准确率的回调函数
const accList = []
const lossList = []
const testAccList = []
const testLossList = []
const epochEndCallback = async (epoch, logs) => {
  console.log(`--Epoch ${epoch + 1}, loss=${logs.loss}, acc=${logs.acc}`);
  accList.push(logs.acc)
  lossList.push(logs.loss)
};

const train = async () => {
  //data
  // const {trainX,trainY,classNames} = await getData(dirPath, outPath)
  const { ds, wasteNamesList } = await getData(dirPath, outPath)
  console.log('ooo')
  const mobileNet = await tf.loadLayersModel(mobileNetPath)
  // mobileNet.summary()
  // console.log(mobileNet.layers.map((layer, i)=>[layer.name, i])) // 0-86
  console.log('111')
  const model = tf.sequential();
  //截断模型，复用了86个层
  for (let i = 0; i <= 86; ++i) {
    const layer = mobileNet.layers[i];
    if (i <= 73) {
      layer.trainable = false;
    }
    model.add(layer);
  }
  //降维，摊平数据
  model.add(tf.layers.flatten());
  //设置全连接层

  model.add(tf.layers.dense({
    units: 60, //62
    activation: 'relu'//设置激活函数，用于处理非线性问题
  }));

  model.add(tf.layers.dense({
    units: wasteNamesList.length,
    activation: 'softmax'//用于多分类问题
  }));
  //设置损失函数，优化器
  model.compile({
    loss: 'sparseCategoricalCrossentropy',
    optimizer: tf.train.adam(),
    metrics: ['acc']
  });
  //训练模型

  // await model.fitDataset(ds, {
  //   epochs: 30,
  //   callbacks: { onEpochEnd: epochEndCallback }
  // }).catch((e) => {
  //   console.log(e.message)
  // })
  for (let epoch = 0; epoch < 40; epoch++) {
    // 训练一个 epoch
    console.log('------epoch:',epoch + 1)
    await model.fitDataset(ds, { epochs: 1 })
    let evalOutput = await model.evaluateDataset(ds)
    testAccList.push(evalOutput[0].dataSync()[0])
    testLossList.push(evalOutput[1].dataSync()[0])
    console.log('test Loss: ' + evalOutput[0].dataSync()[0]);
    console.log('test Accuracy: ' + evalOutput[1].dataSync()[0]);
    // 在每个 epoch 结束后保存模型
    await model.save(`file://${process.cwd()}/${outPath}`);

  }
  // await model.save(`file://${process.cwd()}/${outPath}`);
  console.log('train', accList, lossList)
  console.log('test',testAccList,testLossList)
}
const testFunc = async () => {
  const { ds, wasteNamesList } = await testData(dirPath, outPath)
  let model = await tf.loadLayersModel('http://localhost:3004/output/model.json')
  let labels = JSON.parse(fs.readFileSync(path.join(outPath, 'wasteName.json')))
  model.compile({
    loss: 'sparseCategoricalCrossentropy',
    optimizer: tf.train.adam(),
    metrics: ['acc']
  });
  let evalOutput = await model.evaluateDataset(ds)
  console.log('Loss: ' + evalOutput[0].dataSync()[0]);
  console.log('Accuracy: ' + evalOutput[1].dataSync()[0]);
}
// const train = () => {}
module.exports = train