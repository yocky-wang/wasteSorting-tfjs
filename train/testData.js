const fs = require('fs')
const path = require('path')
const tf = require('@tensorflow/tfjs-node-gpu')

const testNum_1_9 = {
  '其他垃圾-一次性餐具': 71,
  '其他垃圾-化妆品瓶': 45,
  '其他垃圾-卫生纸': 30,
  '其他垃圾-尿片': 30,
  '其他垃圾-污损塑料': 46,
  '其他垃圾-烟蒂': 50,
  '其他垃圾-牙签': 30,
  '其他垃圾-破碎花盆及碟碗': 45,
  '其他垃圾-竹筷': 40,
  '其他垃圾-纸杯': 49,
  '其他垃圾-贝壳': 30,
  '厨余垃圾-剩菜剩饭': 60,
  '厨余垃圾-大骨头': 41,
  '厨余垃圾-水果果皮': 49,
  '厨余垃圾-水果果肉': 78,
  '厨余垃圾-茶叶渣': 53,
  '厨余垃圾-菜梗菜叶': 72,
  '厨余垃圾-落叶': 59,
  '厨余垃圾-蛋壳': 44,
  '厨余垃圾-西餐糕点': 63,
  '厨余垃圾-鱼骨': 45,
  '可回收垃圾-充电宝': 43,
  '可回收垃圾-包': 50,
  '可回收垃圾-塑料玩具': 53,
  '可回收垃圾-塑料碗盆': 44,
  '可回收垃圾-塑料衣架': 48,
  '可回收垃圾-快递纸袋': 30,
  '可回收垃圾-报纸': 41,
  '可回收垃圾-插头电线': 63,
  '可回收垃圾-旧书': 56,
  '可回收垃圾-旧衣服': 44,
  '可回收垃圾-易拉罐': 80,
  '可回收垃圾-枕头': 42,
  '可回收垃圾-毛绒玩具': 57,
  '可回收垃圾-泡沫塑料': 30,
  '可回收垃圾-洗发水瓶': 45,
  '可回收垃圾-牛奶盒等利乐包装': 50,
  '可回收垃圾-玻璃': 30,
  '可回收垃圾-玻璃瓶罐': 63,
  '可回收垃圾-皮鞋': 48,
  '可回收垃圾-砧板': 47,
  '可回收垃圾-纸板箱': 39,
  '可回收垃圾-金属食品罐': 37,
  '可回收垃圾-锅': 50,
  '可回收垃圾-食用油桶': 43,
  '可回收垃圾-饮料瓶': 30,
  '有害垃圾-废弃水银温度计': 30,
  '有害垃圾-废旧灯管灯泡': 30,
  '有害垃圾-电池': 37,
  '有害垃圾-药物': 30,
  '有害垃圾-软膏': 44
}

const testData = async (dirPath, outPath) => {
  let classNames = fs.readdirSync(dirPath)
  // fs.writeFileSync(path.join(outPath,'className.json'), JSON.stringify(classNames))
  wasteNamesList = []
  classNames.forEach((className, classIndex) => {
    wasteNames = fs.readdirSync(path.join(dirPath, className))
    wasteNamesList.push(...wasteNames.map(item => className + '-' + item))
  })
  fs.writeFileSync(path.join(outPath, 'wasteName.json'), JSON.stringify(wasteNamesList))
  console.log(wasteNamesList.length) //51
  let data = []

  let wlist = []

  for (let wasteIndex = 0; wasteIndex < wasteNamesList.length; wasteIndex++) {
    let wasteName = wasteNamesList[wasteIndex]
    fs.readdirSync(path.join(dirPath, wasteName.split('-')[0], wasteName.split('-')[1]))
      .filter(item => item.match(/(jpg)|(jpeg)$/))
      .slice(0, testNum_1_9[wasteName]) // 测试集
      .forEach(filename => {
        // console.log(className,wasteNames[0],filename)
        const imgPath = path.join(dirPath, wasteName.split('-')[0], wasteName.split('-')[1], filename)
        data.push({ imgPath, wasteIndex })
      })
    wlist.push(wasteName.split('-')[1])
    // console.log(wasteName)
  }

  console.log(data.length) //26545

  const ds = tf.data.generator(function* () {
    const count = data.length
    const batchSize = 22
    console.time('batch')
    for (let start = 0; start < count; start += batchSize) {
      const end = Math.min(start + batchSize, count)
      yield tf.tidy(() => {
        const inputs = []
        const labels = []

        for (let j = start; j < end; j++) {
          const { imgPath, wasteIndex } = data[j]
          try {
            const buffer = fs.readFileSync(imgPath)
            const imgTensor = tf.node.decodeJpeg(new Uint8Array(buffer))
            const imgTs = tf.reshape(imgTensor.toFloat().div(127.5).sub(1), [1, 224, 224, 3])
            inputs.push(imgTs)
            labels.push(wasteIndex)
          } catch (e) {
            console.log('error:', e)
            fs.unlinkSync(imgPath)
          }
        }
        const trainX = tf.concat(inputs)
        const trainY = tf.tensor(labels)
        return { xs: trainX, ys: trainY }
      })
    }
    console.timeEnd('batch')
  })
  return {
    ds, wasteNamesList
  }
}

module.exports = testData