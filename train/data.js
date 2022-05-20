const fs = require('fs')
const path = require('path')
const tf = require('@tensorflow/tfjs-node-gpu')

const img_tensor = (imgPath)=>{
  const buffer = fs.readFileSync(imgPath)
  return tf.tidy(()=>{
    const imgTensor = tf.node.decodeImage(new Uint8Array(buffer))
    const newImgTensor = tf.image.resizeBilinear(imgTensor, [224,224])
    // console.log(newImgTensor.toFloat().sub(255/2).div(255/2).size)
    //归一化
    return tf.reshape(newImgTensor.toFloat().sub(255/2).div(255/2), [1,224,224,3])
  })
}
const getData = async (dirPath,outPath)=>{
  let classNames = await fs.readdirSync(dirPath)
  // fs.writeFileSync(path.join(outPath,'className.json'), JSON.stringify(classNames))
  wasteNamesList = []
  classNames.forEach((className,classIndex)=>{
    wasteNames = fs.readdirSync(path.join(dirPath,className))
    wasteNamesList.push(...wasteNames.map(item=>className+'-'+item))
  })
  fs.writeFileSync(path.join(outPath,'wasteName.json'), JSON.stringify(wasteNamesList))
  // console.log(wasteNamesList.length) //60
  let data = []
  wasteNamesList.forEach((wasteName,wasteIndex)=>{
    console.log(path.join(dirPath,wasteName.split('-')[0],wasteName.split('-')[1]))
    let dir = fs.readdirSync(path.join(dirPath,wasteName.split('-')[0],wasteName.split('-')[1]))
    .filter(item=>item.match(/jpg$/))
    .forEach(filename=>{
      // console.log(className,wasteNames[0],filename)
      const imgPath = path.join(dirPath,wasteName.split('-')[0],wasteName.split('-')[1],filename)
      data.push({imgPath, wasteIndex})
    }) 
  })
  console.log(data.length) //27265
  tf.util.shuffle(data) //打乱顺序
  const ds = tf.data.generator(function* (){
    const count = data.length
    const batchSize = 22
    for(let start = 0; start<count;start+=batchSize){
      const end = Math.min(start+batchSize,count)
      yield tf.tidy(()=>{
        const inputs = []
        const labels = []
        for(let j=start; j<end; j++){
          const {imgPath,wasteIndex } = data[j]
          try {
            const imgTs = img_tensor(imgPath)
            inputs.push(imgTs)
            labels.push(wasteIndex)
          } catch (e){
            console.log('error:',imgPath)
            fs.unlinkSync(imgPath)
          }
        }
        const trainX = tf.concat(inputs)
        const trainY = tf.tensor(labels)
        return {xs:trainX,ys:trainY}
      })
    }
  })
  return {
    ds,wasteNamesList
  }
}

module.exports = getData