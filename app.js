const express = require('express')
const train = require('./train')

const app = express()
app.use(express.static('train'))
app.use(express.static('output'))
 
/* ... */

app.listen(3003, () => console.log('Server ready'))
train()
