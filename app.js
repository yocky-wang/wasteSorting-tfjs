const express = require('express')
const test = require('./train')
const other = require('./train/other1')

const app = express()
app.use('/train', express.static('train'))
app.use('/output', express.static('output-5-6'))

/* ... */

// app.listen(3004, () => train())
app.listen(3004, () => test())
// app.listen(3005, () => {other()})
