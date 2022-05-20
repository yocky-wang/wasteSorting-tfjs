const getData = require('./data')
const tf = require('@tensorflow/tfjs-node-gpu')


const dirPath = './垃圾目录'
const outPath = './output'
const mobileNetPath = 'http://localhost:3003/mobile.json'

// const mobileNetPath = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json'
// const mobileNetPath = 'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json'

const train = async ()=>{
  //data
  // const {trainX,trainY,classNames} = await getData(dirPath, outPath)
  const {ds, wasteNamesList} = await getData(dirPath, outPath)
  const mobileNet = await  tf.loadLayersModel(mobileNetPath)
  // mobileNet.summary()
  // console.log(mobileNet.layers.map((layer, i)=>[layer.name, i])) // 0-86
  console.log('111')
  const model = tf.sequential();
  //截断模型，复用了86个层
  for (let i = 0; i <= 86; ++i) {
    const layer = mobileNet.layers[i];
    layer.trainable = false;
    model.add(layer);
  }
  //降维，摊平数据
  model.add(tf.layers.flatten());
  //设置全连接层
  model.add(tf.layers.dense({
    units: 62,
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
    metrics:['acc']
  });
  //训练模型

  await model.fitDataset(ds, { epochs: 20 }).catch((e)=>{
    console.log(e.message)
  })
  await model.save(`file://${process.cwd()}/${outPath}`);
}

module.exports = train