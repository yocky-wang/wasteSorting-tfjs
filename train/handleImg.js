const fs = require('fs')
const path = require('path')
// const tf = require('@tensorflow/tfjs-node-gpu')
const Jimp = require('jimp');

const handleImg = (imgPath) => {
  const buffer = fs.readFileSync(imgPath)
  return tf.tidy(() => {
    try {
      const imgTensor = tf.node.decodeImage(new Uint8Array(buffer))
      const newImgTensor = tf.image.resizeBilinear(imgTensor, [224, 224])
      const newImgPath = 'train-' + imgPath.split('\\').slice(0, -1).join('\\')
      const filename = imgPath.split('\\').pop()
      fs.mkdirSync(newImgPath, { recursive: true })
      // 将处理后的图像保存到磁盘上
      tf.node.encodeJpeg(newImgTensor).then((res) => {
        fs.writeFileSync(newImgPath + '\\' + filename, res)
      }).catch((e) => {
        console.log('handleImgErr', e)
        fs.unlinkSync(imgPath)
      })
    } catch {
      console.log(imgPath)
      fs.unlinkSync(imgPath)
    }
    return null
  })
}

const changeSize = async (dirPath, outPath) => {
  let classNames = await fs.readdirSync(dirPath)
  // fs.writeFileSync(path.join(outPath,'className.json'), JSON.stringify(classNames))
  wasteNamesList = []
  classNames.forEach((className, classIndex) => {
    wasteNames = fs.readdirSync(path.join(dirPath, className))
    wasteNamesList.push(...wasteNames.map(item => className + '-' + item))
  })
  fs.writeFileSync(path.join(outPath, 'wasteName.json'), JSON.stringify(wasteNamesList))
  console.log(wasteNamesList.length) //53
  let data = []
  let nums = {}

  for (let wasteIndex = 40; wasteIndex < wasteNamesList.length; wasteIndex++) {
    let wasteName = wasteNamesList[wasteIndex]
    await new Promise((resolve) => {
      let fileList = fs.readdirSync(path.join(dirPath, wasteName.split('-')[0], wasteName.split('-')[1]))
        .filter(item => item.match(/(jpg)|(jpeg)$/))
      nums[wasteName] = fileList.length
      fileList.forEach(filename => {
        // console.log(className,wasteNames[0],filename)
        const imgPath = path.join(dirPath, wasteName.split('-')[0], wasteName.split('-')[1], filename)
        handleImg(imgPath)
        data.push({ imgPath, wasteIndex })
      })
      resolve()
      console.log(wasteName)
    })
  }
  console.log(nums)
}

// changeSize('./test-垃圾目录', './output')
const getNum = (dirPath) => {
  let classNames = fs.readdirSync(dirPath)
  console.log(classNames.length)
  return classNames.length
}
// getNum('./train-垃圾目录/其他垃圾/尿片')

const getNums = async (dirPath) => {
  let classNames = fs.readdirSync(dirPath).sort((a, b) => b - a)
  let wasteNamesList = []
  let nums = {}
  classNames.forEach((className, classIndex) => {
    wasteNames = fs.readdirSync(path.join(dirPath, className)).sort((a, b) => a - b)
    // wasteNamesList.push(...wasteNames.map(item => className + '-' + item))
    wasteNamesList.push(...wasteNames.map(item => className + '-' + item))
  })
  // console.log(wasteNamesList)
  for (let i in wasteNamesList) {
    let wasteName = wasteNamesList[i]
    let fileList = fs.readdirSync(path.join(dirPath, wasteName.split('-')[0], wasteName.split('-')[1]))
    nums[wasteName] = fileList.length
  }
  console.log(nums)
}
// getNums('./train-垃圾目录')


// 调整亮度和对比度
const adjustBright = (image) => {
  const brightness = Math.random() * 0.5 - 0.25;
  const contrast = Math.random() * 0.5 - 0.25;
  image.brightness(brightness).contrast(contrast);
}
const adjustAngle = (image) => {
  let angel = [90, 180, 270]
  let randomProcess = Math.floor(Math.random() * 3)
  image.rotate(angel[randomProcess])
}
const randomFlip = (image) => {
  let randomProcess = Math.floor(Math.random() * 2)
  if (randomProcess) {
    // 上下翻转
    image.flip(false, true);
  } else {
    image.flip(true, false);
  }
}

const testImgRot = async (filePath,filename) => {
  let [name, type] = filename.split('.')
  Jimp.read(path.join(filePath,filename), (err, img) => {
    if (err) {
      console.log('err', err)
      throw err
    }
    // img.rotate(270)
    img.flip(false, true)
    img.write(path.join(filePath,name+'_5.'+type))
  }
  )
}
testImgRot('./test-垃圾目录/可回收垃圾/泡沫塑料','image_00001.jpg')

const imgRot = async (filePath) => {
  let num = getNum(filePath)
  let imgList = fs.readdirSync(filePath)
  // let imgs = []
  // for (let img of imgList) {
  //   imgs.push(Jimp.read(path.join(filePath, img)))
  // }
  for (let i = 0; i < 300 - num; i++) {
    const randomIndex = Math.floor(Math.random() * 70)
    Jimp.read(path.join(filePath, imgList[randomIndex]), (err, randomImage) => {
      if (err) {
        console.log('err', err)
        throw err
      }
      const randomProcess = Math.floor(Math.random() * 6)
      switch (randomProcess) {
        case 0:
          adjustBright(randomImage);
          break;
        case 1:
          adjustAngle(randomImage);
          break;
        case 2:
          randomFlip(randomImage);
          break;
        case 3:
          adjustBright(randomImage);
          adjustAngle(randomImage);
          break;
        case 4:
          adjustBright(randomImage);
          adjustAngle(randomImage);
          break;
        case 5:
          adjustAngle(randomImage);
          randomFlip(randomImage);
          break;
      }
      randomImage.write(path.join(filePath, 'z_new_image_' + i + '.jpg'));
    })

  }
  // for (let img of imgList.slice(0,1)) {
  //   let [imgName, imgType] = img.split('.')
  //   // console.log(path.join(filePath, img)) //train-垃圾目录\其他垃圾\卫生纸\baidu000000.jpg
  //   Jimp.read(path.join(filePath, img), function(err, image) {
  //     if (err) throw err;

  //     // 旋转图像
  //     image.flip(true, true);
  //     // 保存旋转后的图像
  //     image.write(path.join(filePath, imgName+'_1.'+imgType));
  // });
}
// imgRot('./train-垃圾目录/有害垃圾/药物')
// 其他垃圾-卫生纸 77
// 其他垃圾-尿片 83
// 其他垃圾-牙签 144
// 其他垃圾-贝壳 223
// 可回收垃圾-快递纸袋 280
// 可回收垃圾-泡沫塑料 252
// 可回收垃圾-玻璃 220
// 可回收垃圾-饮料瓶 248
// 有害垃圾-废弃水银温度计 93
// 有害垃圾-废旧灯管灯泡 271
// 有害垃圾-药物 168