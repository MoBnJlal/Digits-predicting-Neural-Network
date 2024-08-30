

const getMethods = (obj) => {
  let properties = new Set()
  let currentObj = obj
  do {
    Object.getOwnPropertyNames(currentObj).map(item => properties.add(item))
  } while ((currentObj = Object.getPrototypeOf(currentObj)))
  return [...properties.keys()].filter(item => typeof obj[item] === 'function')
}
function add(ar1, ar2) {
  let ar = []
  for (var i = 0; i < ar1.length; i++) {
    ar[i] = []
    for (var j = 0; j < ar1[0].length; j++) {
      if(typeof ar2 == 'number') ar[i][j] = ar1[i][j] + ar2
      else {
        if(typeof ar2[0].length == 'number'){
          if(ar2.length == ar1.length)ar[i][j] = ar1[i][j] + ar2[i][j] 
          else ar[i][j] = ar1[i][j] + ar2[0][j]
        }
        if(typeof ar2[0].length != 'number')ar[i][j] = ar1[i][j] + ar2[j]
      }
    }
  }
  return ar 
}  //Done
function sub(ar1, ar2) {
  let ar = []
  for (var i = 0; i < ar1.length; i++) {
    ar[i] = []
    for (var j = 0; j < ar1[0].length; j++) {
      if (typeof ar2 == 'number') ar[i][j] = ar1[i][j] - ar2
      else {
        if (typeof ar2[0].length == 'number') {
          if (ar2.length == ar1.length) ar[i][j] = ar1[i][j] - ar2[i][j]
          else ar[i][j] = ar1[i][j] - ar2[0][j]
        }
        if (typeof ar2[0].length != 'number') ar[i][j] = ar1[i][j] - ar2[j]
      }
    }
  }
  return ar
}  //Done
function mult(ar1, ar2) {
  let ar = [],t
  typeof ar1 == 'number' ? t = 3 : typeof ar1[0].length == 'number' ? t = 1: t = 2 
  for (var i = 0; i < ar2.length; i++) {
    ar[i] = []
    for (var j = 0; j < ar2[0].length; j++) {
      if(t == 1)ar[i][j] = ar1[i][j] * ar2[i][j] 
      if(t == 2)ar[i][j] = ar1[j] * ar2[i][j]
      if(t == 3)ar[i][j] = ar1 * ar2[i][j]
    }
  }
  return ar 
}
function div(ar1, ar2) {
  let ar = [],t
  typeof ar2 == 'number' ? t = 3 : typeof ar2[0].length == 'number' ? t = 1: t = 2 
  for (var i = 0; i < ar1.length; i++) {
    ar[i] = []
    for (var j = 0; j < ar1[0].length; j++) {
      if(t == 1)ar[i][j] = ar1[i][j] / ar2[i][j] 
      if(t == 2)ar[i][j] = ar1[i][j] / ar2[j]
      if(t == 3)ar[i][j] = ar1[i][j] / ar2
    }
  }
  return ar 
}
function dot(ar1, ar2) {
  ar = []
  for (var i = 0; i < ar1.length; i++) {
    ar[i] = []
    for (var j = 0; j < ar2[0].length; j++) {
      s = 0
      for (var k = 0; k < ar2.length; k++) {
        s += ar1[i][k] * ar2[k][j]
      }
     ar[i][j] = s
    }
  }
  return ar
}
function sum(ar, axis = 'non',keepdims = false) {
  if (axis == 'non') {
  sum1 = 0
  ar.forEach((r, i) => {
    sum2 = 0
    //console.log(" r", r)
    /*
    r.forEach((el, j) => {
      sum2 += el
    })
    */
    steps = Math.floor(r.length / 200)
      //console.log("here", train_steps)
    if (steps * 200 < r.length) steps += 1
    for (var s = 0; s < steps; s++) {
      // Tab to edit
      batch = r.slice(s * 200, (s + 1) * 200)
      //console.log("batch No ", s,  batch)
      for (var i = 0; i < batch.length; i++) {
        sum2 += batch[i]
      }    


    }

    /*
    for (var i = 0; i < r.length; i++) {
      sum2 += r[i]
    }*/
    sum1 += sum2
  })
  return sum1
  }
  
  if (axis == 0) {
    ar = T(ar)
    nar = []
    ar.forEach((r, i) => {
      nar[i] = 0
      r.forEach((el, j) => {
        nar[i] += el
      })
    })
  }
  if (axis == 1) {
    nar = []
    ar.forEach((r, i) => {
      nar[i] = 0
      r.forEach((el, j) => {
        nar[i] += el
      })
    })
    if (keepdims) return T([nar])
  }

  return nar 
}
function maximum(x, ar) {
  return ar.map(r => r.map(v => Math.max(...[x, v])))
}
function argmax(ar) {
  return ar.map(r => r.indexOf(Math.max(...r)))
}
function mean(ar, axis = 0) {
  if (axis == "none") {
    s = 0;
    c = 0;
    for (let i = 0; i < ar.length; i++) {
      for (let j = 0; j < ar[i].length; j++) {
        s += ar[i][j];
        c++;
      }
    }
    return s / c;
  }
  if (axis == 0) {
    if (ar[0].length == undefined) {
      s = 0;
      ar.forEach(el => {
        s += el
      })
      return s / ar.length
    } else {
      ar = T(ar)
      nar = []
      ar.forEach(r => {
        s = 0
        r.forEach(e => s += e)
        nar.push(s / r.length)
      })
      return nar
    }
  }
  if(axis == 1) {
    nar = []
    ar.forEach(r => {
      s = 0
      r.forEach(e => s += e)
      nar.push(s / r.length)
    })
    return nar
  }
}
function std(ar) {
    const meanValue = mean(ar, "none");
    let sumSquaredDifferences = 0;
    let count = 0;
    for (let i = 0; i < ar.length; i++) {
        for (let j = 0; j < ar[i].length; j++) {
            const difference = ar[i][j] - meanValue;
            sumSquaredDifferences += difference * difference;
            count++;
        }
    }
    const variance = sumSquaredDifferences / count;
    return Math.sqrt(variance);
}

function clip(ar, l, g) {
  ar.forEach((r, i) => {
    r.forEach((el, j) => {
      if (el < l) ar[i][j] = l
      if (el > g) ar[i][j] = g
    })
  })
  return ar
}
function neg(ar) {
  return ar.map(v => {
    return v.map(el => -el)
  })
}


function pow(ar, n) {
  return ar.map(v => {
    return v.map(el => Math.pow(el, n))
  })
}
function abs(ar) {
  return ar.map(v => {
    return v.map(el => Math.abs(el))
  })
}
function exp(ar) {
  return ar.map(v => {
    return v.map(el => Math.exp(el))
  })
}
function sqrt(ar) {
  return ar.map(v => {
    return v.map(el => Math.sqrt(el))
  })
}
function log(ar) {
  return ar.map(v => {
    return v.map(el => Math.log(el))
  })
}
function sin(ar) {
  return ar.map(v => {
    return v.map(el => Math.sin(el))
  })
}
function cos(ar) {
  return ar.map(v => {
    return v.map(el => Math.cos(el))
  })
}

function zeros(r, c) {
  ar = [];
  for (var i = 0; i < r; i++) {
    ar[i] = [];
    for (var j = 0; j < c; j++) {
      ar[i][j] = 0;
    }
  }
  return ar;
}
function ones(r, c) {
  ar = [];
  for (var i = 0; i < r; i++) {
    ar[i] = [];
    for (var j = 0; j < c; j++) {
      ar[i][j] = 1;
    }
  }
  return ar;
}
function eye(n) {
  ar = [];
  for (var i = 0; i < n; i++) {
    ar[i] = [];
    for (var j = 0; j < n; j++) {
    i == j ?  ar[i][j] = 1 : ar[i][j] = 0;
    }
  }
  return ar;
}
function full(r, c, k) {
  ar = [];
  for (var i = 0; i < r; i++) {
    ar[i] = [];
    for (var j = 0; j < c; j++) {
      if(typeof k == 'number')ar[i][j] = k
      else if (shape(k)[1] == 1)ar[i][j] = k[i][0]
    }
  }
  return ar;
}
function random(r, c) {
  ar = [];
  for (var i = 0; i < r; i++) {
    ar[i] = [];
    for (var j = 0; j < c; j++) {
      ar[i][j] = (Math.random() - 0.5) * 2;
    }
  }
  return ar;
}
function diag(ar) {
  nar = []
  for (var i = 0; i < ar.length; i++) {
    nar.push([])
    for (var j = 0; j < ar.length; j++) {
      i == j ? nar[i][j] = ar[i] : nar[i][j] = 0
    }
  }
  return nar
}
function sign(ar) {
  return ar.map(v => {
    return v.map(el => el > 0 ? 1 : el == 0 ? 0 : -1)
  })
}
function linspace(s, e, st) {
  ar = []
  incr = (e - s) / (st - 1)
  x = s
  for (var i = s; i < s + st; i++) {
    ar.push(x)
    x += incr
  }
  return ar
}
function oneHot(arr, cs) {
  return arr.map(e => {
    a = []
    for (var i = 0; i < cs; i++) {
    a[i] = e == i ? 1 : 0
    }
    return a
  })
}

function T(ar) {
  nar = []
  for (var i = 0; i < ar[0].length; i++) {
    nar[i] = []
    for (var j = 0; j < ar.length; j++) {
      nar[i][j] = ar[j][i]
    }
  }
  return nar
}
function shape(ar) {
  if(typeof ar == 'number') return 0
  if(ar[0].length == undefined) return [ar.length] 
  return [ar.length, ar[0].length]
}
function indices(ar,ix) {
  return ar.map((v,i) => v[ix[i]])
}
function copy(ar) {
  if (typeof ar != 'object') return JSON.parse(JSON.stringify([ar]))[0]
  return JSON.parse(JSON.stringify(ar))
}
function show(ar, trm = 0, pr = 3) {
  if (trm == 0){
  artx = ''
  ar.forEach((r, i) => {
    r.forEach((el, j) => {
      artx += '   '
      artx += Math.round(el * 10**pr) / 10**pr
    })
    artx += '\n'
  })
  console.log(artx)
  }
  if (trm == 1) {
    table = document.createElement('table')
    ar.forEach((r, i) => {
      row = document.createElement('tr')
      r.forEach((el, j) => {
        c = document.createElement('td')
        c.innerText = Math.round(el * 10**pr) / 10**pr
        row.appendChild(c)
      })
      table.appendChild(row);
    })
    document.getElementById('output').appendChild(table);
  }
}
function check(ar1,ar2,op) {
  return ar1.map((_, i) => op(ar1[i], ar2[i]))
}

function spiral_data(samples, classes) {
  xv = []
  yv = []
  for (var i = 0; i < classes; i++) {
    r = linspace(0, 1, samples)
    t = add(mult(0.3, random(1, samples)), linspace(i*4, (i+1)*4, samples))
    sn = mult(r, sin(mult(2.5, t)))[0]
    cs = mult(r, cos(mult(2.5, t)))[0]
    for (var k = 0; k < samples; k++) {
      xv.push([sn[k], cs[k]])
      yv.push(i)
    }
  }
  return [xv, yv]
}
function sine_data(samples = 1000) {
  xv = []
  yv = []
  for (var i = 0; i < samples; i++) {
    xv.push([i/samples])
    yv.push([Math.sin(-2*Math.PI*(i/samples))])
  }
  return [xv, yv]
}
function scatter(cords, xo = 0, yo = 0) {
  for (var i = 0; i < cords[0].length; i++) {
    ctx.beginPath()
    //tonsole.log(i)
    ctx.arc((cords[0][i][0] + 1) * 120 + xo, (cords[0][i][1] + 1) * 120 + yo, 2, 0, Math.PI * 2);
   //ctx.fillStyle = 'blue'
    ctx.fillStyle = ['#C90000', '#0A8100', '#002DA2'][cords[1][i]]
    ctx.fill()
  }
  ctx.fillStyle = 'blue'
}
function line(cords, xo = 0, yo = 0) {
  for (var i = 0; i < cords[0].length; i++) {
    ctx.beginPath()
    //tonsole.log(i)
    ctx.arc((cords[0][i][0]) * 240 + xo, (cords[1][i][0] + 1) * 120 + yo, 0.75, 0, Math.PI * 2);
    ctx.fillStyle = 'blue'
    //ctx.fillStyle = '#002D88'
    ctx.fill()
  }
  ctx.fillStyle = 'blue'
}


function plot(dim, px, py, model) {
  //ctx.clearRect(250, 0, 250, 250)
  for (var i = 0; i <= dim; i++) {
    for (var j = 0; j <= dim; j++) {
      ctx.beginPath()
      ctx.rect((i - dim / 2) *0.99* 250/dim + px, (j - dim / 2) *0.99* 250/dim + py, 250/dim, 250/dim);
      //console.log((i/dim - 0.5)*2, (j/dim - 0.5)*2)
      let output = model.forward([[(i / dim - 0.5) * 2, (j / dim - 0.5) * 2]], false)
      //console.log(output)
      let predictions = model.output_layer_activation.predictions(output)
      //console.log("predictions", predictions)

      /*
      dense1.forward([[(i / dim - 0.5) * 2, (j / dim - 0.5) * 2]])
      activation1.forward(dense1.output)
      dense2.forward(activation1.output)
      loss_activation.forward(dense2.output, y)
      */
      
      ctx.fillStyle = 'rgb(' +
        Math.round(output[0][0] * 255) + ',' +
        Math.round(output[0][1] * 255) + ',' +
        Math.round(output[0][2] * 255) + ')'
      ctx.fill()
    }
  }
  //ctx.closePath()
  ctx.fillStyle= 'blue'
}
function plotB(dim, px, py) {
  //ctx.clearRect(250, 0, 250, 250)
  for (var i = 0; i <= dim; i++) {
    for (var j = 0; j <= dim; j++) {
      ctx.beginPath()
      ctx.rect((i - dim / 2) * 0.99 * 250 / dim + px, (j - dim / 2) * 0.99 * 250 / dim + py, 250 / dim, 250 / dim);
      //console.log((i/dim - 0.5)*2, (j/dim - 0.5)*2)
      
      dense1.forward([[(i / dim - 0.5) * 2, (j / dim - 0.5) * 2]])
      activation1.forward(dense1.output)
      dense2.forward(activation1.output)
      activation2.forward(dense2.output)
      predictions = activation2.output.map(r => r.map(e => e > 0.5 ? 1 : 0))
      
      
      ctx.fillStyle = 'rgb(' +
        predictions[0][0] * 255 + ',' +
        predictions[0][0] * 255 + ',' +
        predictions[0][0] * 255 + ')'
        
      ctx.fill()
      //ctx.closePath()
      ctx.fillStyle = 'blue'
    }
  }
  //ctx.closePath()
 // ctx.fillStyle = 'blue'
}
function potB(dim, px, py) {
  //ctx.clearRect(250, 0, 250, 250)
  for (var i = 0; i <= dim; i++) {
    for (var j = 0; j <= dim; j++) {
      //console.log(i,j)
      ctx.beginPath()
      ctx.rect((i - dim / 2) * 0.99 * 250 / dim + px, (j - dim / 2) * 0.99 * 250 / dim + py, 250 / dim, 250 / dim);
      //console.log((i/dim - 0.5)*2, (j/dim - 0.5)*2)

      dense1.forward([[(i / dim - 0.5) * 2, (j / dim - 0.5) * 2]])
      activation1.forward(dense1.output)
      dense2.forward(activation1.output)
      activation2.forward(dense2.output)
      predictions = activation2.output.map(r => r.map(e => e > 0.5 ? 1 : 0))

      //console.log('predictions', predictions[0][0])

      ctx.fillStyle = 'rgb(' +
        predictions[0][0] * 255 + ',' +
        predictions[0][0] * 255 + ',' +
        predictions[0][0] * 255 + ')'

      ctx.fill()
      //ctx.closePath()
      ctx.fillStyle = 'blue'
    }
  }
  //ctx.closePath()
  // ctx.fillStyle = 'blue'
}
function plotL(dim, px, py) {
  ctx.clearRect(250, 0, 250, 250)
  for (var i = 0; i < dim; i++) {
    //console.log(i/dim)
    dense1.forward([[i / dim]])
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    predictions = activation3.output
    //console.log('pred', predictions)
    //console.log('--------++++-')
    ctx.beginPath()
    //tonsole.log(i)
    ctx.arc( i / dim * 240 + px - 120, (predictions[0][0] + 1) * 120 + 0, 0.75, 0, Math.PI * 2);
    ctx.fillStyle = 'red'
    //ctx.fillStyle = '#002D88'
    ctx.fill()
  }
  ctx.fillStyle = 'blue'
}
function roundTo(value, places) {
  var pow = Math.pow(10, places);
  if (pow < 1) return Math.round(value * pow)
  return Math.round(value * pow) / pow;
}


/*

function reduceMatrix(matrix, newRows, newCols) {
    const oldRows = matrix.length;
    const oldCols = matrix[0].length;

    // Calculate block size for each dimension
    const rowBlockSize = Math.floor(oldRows / newRows);
    const colBlockSize = Math.floor(oldCols / newCols);

    const newMatrix = Array.from({ length: newRows }, () => Array(newCols).fill(0));

    for (let newRow = 0; newRow < newRows; newRow++) {
        for (let newCol = 0; newCol < newCols; newCol++) {
            let sum = 0;
            let count = 0;

            // Calculate the average value of the block
            for (let r = newRow * rowBlockSize; r < (newRow + 1) * rowBlockSize && r < oldRows; r++) {
                for (let c = newCol * colBlockSize; c < (newCol + 1) * colBlockSize && c < oldCols; c++) {
                    sum += matrix[r][c];
                    count++;
                }
            }

            // Set the average value in the new matrix
            newMatrix[newRow][newCol] = sum / count;
        }
    }

    return newMatrix;
}

// Example usage:
const matrix = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]

];
const reducedMatrix = reduceMatrix(matrix, 3, 3)
show(reducedMatrix);
document.getElementById('upload').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(event) {
                const img = new Image();
                img.onload = function() {
                    resizeImage(img, 60, 60); // Set desired width and height
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        });

        function resizeImage(image, width, height) {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            canvas.width = width;
            canvas.height = height;

            // Set image smoothing quality
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high'; // Can be 'low', 'medium', or 'high'

            // Draw the image resized
            ctx.drawImage(image, 0, 0, width, height);
            console.log(ctx.getImageData(0, 0, width, height).data.length)
        }
*/

