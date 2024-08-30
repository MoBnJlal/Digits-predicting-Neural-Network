class Model {
  constructor() {
    this.layers = []
    this.softmax_classifier_output = "None"
  }
  add(layer) {
    this.layers.push(layer)
  }
  set (loss = 'None', optimizer = 'None', accuracy = 'None'){
    if(loss != 'None')this.loss = loss
    if(optimizer != 'None')this.optimizer = optimizer
    if(accuracy != 'None')this.accuracy = accuracy
  }
  finalize(){
    this.input_layer = new LI()
    let layer_count = this.layers.length
    this.trainable_layers = []

    for (var i = 0; i < layer_count; i++) {
      if (i == 0) {
        this.layers[i].prev = this.input_layer                 
        this.layers[i].next = this.layers[i+1]
      } else if (i < layer_count - 1) {
        this.layers[i].prev = this.layers[i-1]                 
        this.layers[i].next = this.layers[i+1]
      } else {
        this.layers[i].prev = this.layers[i-1]                 
        this.layers[i].next = this.loss
        this.output_layer_activation = this.layers[i]
      }
      if (this.layers[i].hasOwnProperty('weights')) {
        this.trainable_layers.push(this.layers[i])
      }
    }
    
    if(this.loss != 'None')this.loss.remember_trainable_layers(this.trainable_layers)

    if((this.layers[this.layers.length -1] instanceof Act_Softmax && this.loss instanceof Loss_CCE)){
      this.softmax_classifier_output = new Act_Softmax_Loss_CCE() 
    }  
  }
  train(X, y, epochs = 1000, batch_size = "None",  print_every = 1, validation_data = "None") {
    this.accuracy.init(y)
    let train_steps = 1
    if (validation_data != "None"){
      let validation_steps = 1
      let [X_val, y_val] = validation_data
    }
    if (batch_size != "None"){
      train_steps = Math.floor(shape(X)[0] / batch_size )
      if (train_steps * batch_size < shape(X)[0]) train_steps += 1;
    }
    let epoch = 1
    let md = this
    ctx.fillStyle = 'blue'
    requestAnimationFrame(animate);
    //for (var epoch = 1; epoch < epochs + 1; epoch++) {
    function animate() {
      md.loss.new_pass()
      md.accuracy.new_pass()
      
      for (var step = 0; step < train_steps; step++) {
        let batch_X, batch_y
        if(batch_size == "None"){
          batch_X = X
          batch_y = y
        } else {
          batch_X = X.slice(step*batch_size,(step+1)*batch_size)
          batch_y = y.slice(step*batch_size,(step+1)*batch_size)
        }
        let output = md.forward(batch_X, true)
        let [data_loss, reg_loss] = md.loss.calculate(output, batch_y, true)
        let loss = data_loss + reg_loss
        let predictions = md.output_layer_activation.predictions(output)
        accuracy = md.accuracy.calculate(predictions, batch_y)
        md.backward(output, batch_y)
        md.optimizer.pre_update_params()
        for (let layer of md.trainable_layers) {
          md.optimizer.update_params(layer)
        }
        md.optimizer.post_update_params()
        if (false) {
          ctx.clearRect(0, 267, 500, 100);
          ctx.fillText("step: " + step, 10, 275)
          ctx.fillText("acc: " + roundTo(accuracy, 3), 10, 290)
          ctx.fillText("loss: " + roundTo(loss, 5), 10, 305)
          ctx.fillText("data loss: " + roundTo(data_loss, 5), 10, 320)
          ctx.fillText("reg_loss: " + roundTo(reg_loss, 5), 10, 335)
          ctx.fillText("lr: " + roundTo(optimizer.current_learning_rate, 4), 10, 350)
        }
        requestAnimationFrame(()=>{
          })

      }
      
      let [ep_data_loss, ep_reg_loss] = md.loss.calculate_accumulated(true)
      let ep_loss = ep_data_loss + ep_reg_loss
      let ep_accuracy = md.accuracy.calculate_accumulated()
      ctx.clearRect(0, 267, 500, 100);
      ctx.fillText("epoch: " + epoch, 10, 275)
      ctx.fillText("acc: " + roundTo(ep_accuracy, 3), 10, 290)
      ctx.fillText("loss: " + roundTo(ep_loss, 5), 10, 305)
      ctx.fillText("data loss: " + roundTo(ep_data_loss, 5), 10, 320)
      ctx.fillText("reg_loss: " + roundTo(ep_reg_loss, 5), 10, 335)
      ctx.fillText("lr: " + roundTo(optimizer.current_learning_rate, 4), 10, 350)
      if (validation_data != "None") {
        let [X_val, y_val] = validation_data
        md.evaluate(X_val, y_val, batch_size)
      }
      
      epoch++
      if (epoch < epochs + 1) requestAnimationFrame(animate)
      else {
        md.save('fashion_mnist.model')
        return
        let [X_test, y_test] = validation_data
        let prams = md.get_parameters()
        
        let model = new Model()
        model.add(new LD(shape(X)[1], 64))
        model.add(new Act_ReLU())
        model.add(new LD(64, 64))
        model.add(new Act_ReLU())
        model.add(new LD(64, 10))
        model.add(new Act_Softmax())
        model.set(loss = new Loss_CCE(), 'None', accuracy = new Accuracy_Categorical())
        model.finalize()
        model.set_parameters(prams)
      }
    }
  }
  evaluate(X_val,y_val,batch_size = 'None',sh = false){
    let validation_steps = 1
    let batch_X, batch_y
    if (batch_size != "None") {
      validation_steps = Math.floor(shape(X_val)[0] / batch_size)
      if (validation_steps * batch_size < shape(X_val)[0]) validation_steps += 1
    }
    this.loss.new_pass()

    this.accuracy.new_pass()

    for (var step = 0; step < validation_steps; step++) {
      if (batch_size == "None") {
        //console.log("")
        batch_X = X_val
        batch_y = y_val
      } else {
        batch_X = X_val.slice(step * batch_size, (step + 1) * batch_size)
        batch_y = y_val.slice(step * batch_size, (step + 1) * batch_size)
      }
      
      
      let output = this.forward(batch_X, false)
      this.loss.calculate(output, batch_y)
      let predictions = this.output_layer_activation.predictions(output)
      this.accuracy.calculate(predictions, batch_y)
    }
    let val_loss = this.loss.calculate_accumulated()
    
    let val_accuracy = this.accuracy.calculate_accumulated()

    return [val_accuracy, val_loss]
  }
  predict(X, batch_size = "None"){
    let prediction_steps = 1 
    let batch_X
    if (batch_size != "None") {
      prediction_steps = Math.floor(shape(X)[0] / batch_size)
      if (prediction_steps * batch_size < shape(X)[0]) prediction_steps += 1
    }
    let output = []
    for (var step = 0; step < prediction_steps; step++) {
      if (batch_size == "None") {
        batch_X = X
      } else {
        batch_X = X.slice(step * batch_size, (step + 1) * batch_size)
      }
      
    }
    let batch_output = this.forward(batch_X, false)
    output.push(batch_output)
    return output
  }
  forward(X, training){
    this.input_layer.forward(X, training)
    for (var layer of this.layers){
      layer.forward(layer.prev.output, training) 
    }   
    return layer.output
  }
  backward(output, y) {
    if (this.softmax_classifier_output != "None") {
      this.softmax_classifier_output.backward(output, y)
      this.layers[this.layers.length - 1].dinputs = this.softmax_classifier_output.dinputs
      for (let layer of this.layers.slice(0, -1).reverse()) {
        layer.backward(layer.next.dinputs)
      }
      return
    }
    this.loss.backward(output, y)
    for (let layer of [...this.layers].reverse()) {
      layer.backward(layer.next.dinputs)
    }
  }
  get_parameters(){
    let parameters = []
    for (let layer of this.trainable_layers)parameters.push(layer.get_parameters())
    return parameters
  }
  set_parameters(parameters){
    for (var i in parameters) {
      this.trainable_layers[i].set_parameters(parameters[i][0], parameters[i][1])
    }
  }
  save_parameters(path){
    var json = JSON.stringify(this.get_parameters());
    console.log(path)
    json = [json];
    var blob1 = new Blob(json, { type: "text/plain;charset=utf-8" });
    var isIE = false || !!document.documentMode;
    if (isIE) {
      window.navigator.msSaveBlob(blob1, path);
    } else {
      var url = window.URL || window.webkitURL;
      let link = url.createObjectURL(blob1);
      var a = document.createElement("a");
      a.download = path;
      a.href = link;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }
  load_parameters(path){
    fetch("./"+path)
      .then(response => {
        return response.json();
      })
      .then(data => {
        this.set_parameters(data)
      })
  }
  save(path){
   let md = this
   md.loss.new_pass()
   md.accuracy.new_pass()
   delete md.input_layer.output
   delete md.loss.dinputs
   for (var layer of md.layers) {
     for (let prop of ['inputs', 'output', 'dinputs', 'dweights', 'dbiases', 'next', 'prev']) {
       delete layer[prop]
     }
   }
   let json = [JSON.stringify(md)];
   let blob1 = new Blob(json, { type: "text/plain;charset=utf-8" });
   
   var isIE = false || !!document.documentMode;
   if (isIE) {
     window.navigator.msSaveBlob(blob1, path);
   } else {
     var url = window.URL || window.webkitURL;
     let link = url.createObjectURL(blob1);
     var a = document.createElement("a");
     a.download = path;
     a.href = link;
     document.body.appendChild(a);
     a.click();
     document.body.removeChild(a);
   }
  }
  async load(path){
    const res = await  fetch("./" + path)
      .then(response => {
        return response.json();
      })
    for (var i = 0; i < res.layers.length; i++) {
      let t
      if (res.layers[i].name == "LD") {
        t = new LD()
        t.weights = res.layers[i]. weights
        t.biases = res.layers[i]. biases
        this.layers[i] = t
      } else {
        switch (res.layers[i].name) {
          case 'Act_ReLU':
            t = new Act_ReLU()
            break;
          case 'Act_Softmax':
            t = new Act_Softmax()
            break;
          case 'Act_Sigmoid':
            t = new Act_Sigmoid()
            break;          
          case 'Act_Linear':
            t = new Act_Linear()
            break;          
        }
        this.layers[i] = t
      }
    }
    switch (res.loss.name) {
      case 'Loss_CCE':
        this.loss = new Loss_CCE()
        break;
      case 'Loss_BCE':
        this.loss = new Loss_BCE()
        break;
      case 'Loss_MSE':
        this.loss = new Loss_MSE()
        break;
      case 'Loss_MAE':
        this.loss = new Loss_MAE()
        break;      
    }
    if (res.optimizer.name == "Optimizer_Adam") {
      this.optimizer = new Optimizer_Adam()
      for (let prop of Object.getOwnPropertyNames(new Optimizer_Adam())) {
        this.optimizer[prop] = res.optimizer[prop]
      }
      this.optimizer.current_learning_rate = this.optimizer.learning_rate
       this.optimizer.iterations = 0
    }
    
    switch (res.accuracy.name) {
      case 'Accuracy_Categorical':
        this.accuracy = new Accuracy_Categorical()
        break;
      case 'Accuracy_Regression':
        this.accuracy = new Accuracy_Regression()
        break; 
    }  
    this.finalize()
  }
}
class LI {
  forward(inputs) {
    this.name = "LI"
    this.output = inputs
  }
}  //Input Layer
class LD {
  constructor(n_inputs, n_neurons, w_reg_l1 = 0, w_reg_l2 = 0, b_reg_l1 = 0, b_reg_l2 = 0) {
    this.name = "LD"
    this.weights = mult(0.01, random(n_inputs, n_neurons))
    this.biases = zeros(1, n_neurons);

    this.w_reg_l1 = w_reg_l1
    this.w_reg_l2 = w_reg_l2
    this.b_reg_l1 = b_reg_l1
    this.b_reg_l2 = b_reg_l2
  }
  
  forward(inputs) {
    this.output = add(dot(inputs, this.weights), this.biases) 
    this.inputs = inputs
  }
  
  backward(dvalues) {
    this.dweights = dot(T(this.inputs), dvalues)
    this.dbiases = [sum(dvalues, 0)]
    if (this.w_reg_l1 > 0) {
      let dL1 = this.weights.map(r => r.map(e => e < 0 ? -1 : 1))
      this.dweights = add(this.dweights, mult(this.w_reg_l1, dL1))
    }

    if (this.w_reg_l2 > 0) {
      this.dweights = add(this.dweights, mult(2 * this.w_reg_l2, this.weights))
    }

    if (this.b_reg_l1 > 0) {
      let dL1 = this.biases.map(r => r.map(e => e < 0 ? -1 : 1))
      this.dbiases = add(this.dbiases, mult(this.b_reg_l1, dL1))
    }

    if (this.b_reg_l2 > 0) {
      this.dbiases = add(this.dbiases, mult(2 * this.b_reg_l2, this.biases))
    }

    this.dinputs = dot(dvalues, T(this.weights))
  }
  get_parameters(){
      return [this.weights, this.biases]
  }
  set_parameters(weights, biases){
    this.weights = weights
    this.biases = biases
  }
}  //Dense Layer
class Layer_Dropout {
  constructor(rate) {
    this.name = "Layer_Dropout"
    this.rate = 1 - rate
  }
  forward(inputs, training) {
    this.inputs = inputs
    if (!training) {
      this.output = copy(inputs)
      return
    }
    let mask = ones(inputs.length, inputs[0].length)
    for (var i = 0; i < mask.length; i++) {
      while (true) {
        let index = Math.floor(Math.random() * mask[0].length)
        mask[i][index] = 0
        let dropped_out = 0
        for (let value of mask[i]) {
          if (value == 0) {
            dropped_out += 1
          }
        }
        if (dropped_out / mask[0].length >= 1 - this.rate) {
          break
        }
      }
    }
    this.binary_mask = div(mask, this.rate)

    this.output = mult(inputs, this.binary_mask)
  }
  backward(dvalues) {
    this.dinputs = mult(dvalues, this.binary_mask)
  }
} //Dropout
class Act_ReLU {
  constructor(){
    this.name = "Act_ReLU"
  }
  forward(inputs) {
    this.inputs = inputs
    this.output = maximum(0, inputs)
  }
  backward(dvalues) {
    this.dinputs = mult(this.inputs.map(r => r.map(e => e > 0?1:0)), dvalues)
  }
  predictions(outputs) {
  return outputs
  }
}  //ReLU Activation
class Act_Softmax {
  constructor() {
    this.name = "Act_Softmax"
  }
  forward(inputs) {
    let exp_values = exp(sub(inputs, full(inputs.length, inputs[0].length, sum(inputs, 1, true))))
    let probabilities = div(exp_values, full(exp_values.length, exp_values[0].length, sum(exp_values, 1, true)))
    this.output = probabilities 
  }
  backward(dvalues) {
    this.dinputs = []
    for (var i in this.output) {
      let single_output = T([this.output[i]])
      let jacobian_matrix = sub(diag(single_output), dot(single_output, T(single_output)))
      this.dinputs[i] = dot(jacobian_matrix, dvalues[i])[0]
    }
  }
  predictions(outputs) {
    return argmax(outputs)
  }
} //not completed
class Act_Sigmoid {
  constructor() {
    this.name = "Act_Sigmoid"
  }
  forward(inputs){
    this.inputs = inputs
    this.output = div(ones(inputs.length, inputs[0].length), add(exp(neg(inputs)), 1))
  }
  backward(dvalues){
    this.dinputs = mult(dvalues, mult(add(neg(this.output), 1), this.output))
  }
  predictions(outputs){
    return outputs.map(r => r.map(e => e > 0.5 ? 1 : 0))
  }
} //Sigmoid Activation
class Act_Linear {
  constructor() {
    this.name = "Act_Linear"
  }
  forward(inputs){
    this.inputs = inputs
    this.output = inputs
  }
  backward(dvalues){
    this.dinputs = copy(dvalues)
  }
  predictions(outputs) {
    return outputs
  }
}  //Linear Activation
class Loss {
  regularization_loss() {
    let reg_loss = 0
    for (let layer of this.trainable_layers) {
      if (layer.w_reg_l1 > 0) {
        reg_loss += layer.w_reg_l1 * sum(abs(layer.weights))
      }
      if (layer.w_reg_l2 > 0) {
        reg_loss += layer.w_reg_l2 * sum(mult(layer.weights, layer.weights))
      }
  
      if (layer.b_reg_l1 > 0) {
        reg_loss += layer.b_reg_l1 * sum(abs(layer.biases))
      }
      if (layer.b_reg_l2 > 0) {
        reg_loss += layer.b_reg_l2 * sum(mult(layer.biases, layer.biases))
      }
    }
    return reg_loss
  }
  remember_trainable_layers(trainable_layers){      
    this.trainable_layers = trainable_layers
  }
  calculate(output, y, include_reg = false) {
    let sample_losses = this.forward(output, y)
    L = sample_losses
    console.log("OUTPUT", output[432])
    let data_loss = -mean(sample_losses)
    console.log("before ", this.accumulated_sum)
    this.accumulated_sum += sum([sample_losses])   
    console.log("sample_losses ", sample_losses)
    console.log("after ", this.accumulated_sum)
    this.accumulated_count += sample_losses.length
    if(!include_reg)return data_loss
    return [data_loss, this.regularization_loss()]
  }
  calculate_accumulated(include_reg = false){
  let data_loss = -this.accumulated_sum / this.accumulated_count   
  console.log("---", this.accumulated_sum)
  if (!include_reg)return data_loss     
  return [data_loss, this.regularization_loss()]
  }
  new_pass(){
    this.accumulated_sum = 0
    this.accumulated_count = 0
  }
}  //Loss
class Loss_CCE extends Loss {
  constructor() {
    super()
    this.name = "Loss_CCE"
  }
  forward(y_pred, y_true) {
    let y_pred_clipped = clip(y_pred, 1e-7, 1 - 1e-7)
    let correct_confidences
    if (shape(y_true).length == 1)
      correct_confidences = indices(y_pred_clipped, y_true)
    else if (shape(y_true).length == 2) 
      correct_confidences = sum(mult(y_pred_clipped, y_true), 1)
    
    let log_likelihoods = (log([correct_confidences]))
    return log_likelihoods[0]
  }
  backward(dvalues, y_true) {
    let samples = dvalues.length
    let labels = dvalues[0].length
    if (shape(y_true).length == 1) {
      y_true = oneHot(y_true, labels)
    }
    this.dinputs = div(mult(-1, y_true), dvalues)
    this.dinputs = div(this.dinputs, samples)
    show(this.dinputs)
    this.dinputs = this.dinputs
  }
} //Categorical Cross-entropy loss
class Loss_BCE extends Loss {
  constructor() {
    super()
    this.name = "Loss_BCE"
  }
  forward(y_pred, y_true) {
    let y_pred_clipped = clip(y_pred, 1e-7, 1 - 1e-7)
    let sample_losses = add(mult(y_true, log(y_pred_clipped)), mult(add(neg(y_true), 1), log(add(neg(y_pred_clipped), 1))))
    sample_losses = sample_losses.map(r => mean(r))
    return sample_losses
  }
  backward(dvalues, y_true) {
    let samples = dvalues.length
    let outputs = dvalues[0].length     
    let clipped_dvalues = clip(dvalues, 1e-7, 1 - 1e-7)
    this.dinputs = div(neg(sub(div(y_true, clipped_dvalues), div(add(neg(y_true), 1), add(neg(clipped_dvalues), 1)))), outputs)
    this.dinputs = div(this.dinputs, samples)
  }
} //Binary Cross-entropy loss
class Loss_MSE extends Loss {
  constructor() {
    super()
    this.name = "Loss_MSE"
  }
  forward(y_pred, y_true) {
    //let sample_losses = mean((y_true - y_pred)**2, 1)
    let sample_losses = mean(pow(sub(y_true, y_pred), 2), 1)
    return sample_losses
  }
  backward(dvalues, y_true) {
    let samples = dvalues.length
    let outputs = dvalues[0].length
    this.dinputs = div(mult(-2, sub(y_true, dvalues)), outputs)
    this.dinputs = div(this.dinputs, samples)
  }
} //Mean Squared Error loss
class Loss_MAE extends Loss {
  constructor() {
    super()
    this.name = "Loss_MAE"
  }
  forward(y_pred, y_true) {
    let sample_losses = mean(abs(sub(y_true, y_prec)), 1)
    return sample_losses
  }
  backward(dvalues, y_true) {
    let samples = dvalues.length
    let outputs = dvalues[0].length
    this.dinputs = div(sub(y_true, dvalues), outputs)
    this.dinputs = div(this.dinputs, samples)
  }
} //Mean Absolute Error loss
class Act_Softmax_Loss_CCE {
  constructor() {
    this.activation = new Act_Softmax()
    this.loss = new Loss_CCE()
  }
  forward(inputs, y_true) {
    this.activation.forward(inputs)
    this.output = this.activation.output
    return this.loss.calculate(this.output, y_true)
  }
  backward(dvalues, y_true) {
    let samples = dvalues.length
    let labels = dvalues[0].length
    if (shape(y_true).length == 1) {
      y_true = oneHot(y_true, labels)
    }
    this.dinputs = add(copy(dvalues), mult(-1, y_true))
    this.dinputs = div(this.dinputs, samples)
  }
}  //combined Softmax activation and Cross-entropy loss
class Optimizer_SGD {
  constructor(learning_rate = 1, decay = 0, momentum = 0) {
    this.name = "Optimizer_SGD"
    this.learning_rate = learning_rate
    this.current_learning_rate = learning_rate
    this.decay = decay
    this.iterations = 0
    this.momentum = momentum
  }
  pre_update_params() {
    if (this.decay) {
      this.current_learning_rate = this.learning_rate * (1 / (1 + this.decay * this.iterations))
    }
  }
  update_params(layer) {
    let weight_updates, bias_updates
    if (this.momentum) {
      if (!layer.hasOwnProperty('weight_momentums')) {
        layer.weight_momentums = zeros(layer.weights.length, layer.weights[0].length)
        layer.bias_momentums = zeros(layer.biases.length, layer.biases[0].length)
      }
      weight_updates = sub(mult(this.momentum, layer.weight_momentums), mult(this.current_learning_rate, layer.dweights))
      layer.weight_momentums = weight_updates
      
      bias_updates = sub(mult(this.momentum, layer.bias_momentums), mult(this.current_learning_rate, layer.dbiases))
      layer.bias_momentums = bias_updates
    }
    else {
      weight_updates = mult(-this.current_learning_rate, layer.dweights)
      bias_updates = mult(-this.current_learning_rate, layer.dbiases)
    }
    layer.weights = add(layer.weights, weight_updates)
    layer.biases = add(layer.biases, bias_updates)
  }
  post_update_params() {
    this.iterations += 1
  }
} //SGD optimizer
class Optimizer_Adagrad {
  constructor(learning_rate = 1, decay = 0, epsilon = 1e-7) {
    this.name = "Optimizer_Adagrad"
    this.learning_rate = learning_rate
    this.current_learning_rate = learning_rate
    this.decay = decay
    this.iterations = 0
    this.epsilon = epsilon
  }
  pre_update_params() {
    if (this.decay) {
      this.current_learning_rate = this.learning_rate * (1 / (1 + this.decay * this.iterations))
    }
  }
  update_params(layer) {
    if (!layer.hasOwnProperty('weight_cache')) {
      layer.weight_cache = zeros(layer.weights.length, layer.weights[0].length)
      layer.bias_cache = zeros(layer.biases.length, layer.biases[0].length)
    }

    layer.weight_cache = add(layer.weight_cache, pow(layer.dweights, 2))
    layer.bias_cache = add(layer.bias_cache, pow(layer.dbiases, 2))
    
    layer.weights = add(layer.weights, div(mult(-this.current_learning_rate, layer.dweights), add(sqrt(layer.weight_cache), this.epsilon)))
    layer.biases = add(layer.biases, div(mult(-this.current_learning_rate, layer.dbiases), add(sqrt(layer.bias_cache), this.epsilon)))
  }
  post_update_params() {
    this.iterations += 1
  }
} //Adagrad optimizer
class Optimizer_RMSprop {
  constructor(learning_rate = 0.001, decay = 0, epsilon = 1e-7, rho = 0.9) {
    this.name = "Optimizer_RMSprop"
    this.learning_rate = learning_rate
    this.current_learning_rate = learning_rate
    this.decay = decay
    this.iterations = 0
    this.epsilon = epsilon
    this.rho = rho
  }
  pre_update_params() {
    if (this.decay) {
      this.current_learning_rate = this.learning_rate * (1 / (1 + this.decay * this.iterations))
    }
  }
  update_params(layer) {
    if (!layer.hasOwnProperty('weight_cache')) {
      layer.weight_cache = zeros(layer.weights.length, layer.weights[0].length)
      layer.bias_cache = zeros(layer.biases.length, layer.biases[0].length)
    }
    layer.weight_cache = add(mult(this.rho, layer.weight_cache), mult((1 - this.rho), pow(layer.dweights, 2)))
    layer.bias_cache = add(mult(this.rho, layer.bias_cache), mult((1 - this.rho), pow(layer.dbiases, 2)))

    layer.weights = add(layer.weights, div(mult(-this.current_learning_rate, layer.dweights), add(sqrt(layer.weight_cache), this.epsilon)))
    layer.biases = add(layer.biases, div(mult(-this.current_learning_rate, layer.dbiases), add(sqrt(layer.bias_cache), this.epsilon)))
  }
  post_update_params() {
    this.iterations += 1
  }
} //RMSprop optimizer
class Optimizer_Adam {
  constructor(learning_rate = 0.001, decay = 0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999) {
    this.name = "Optimizer_Adam"
    this.learning_rate = learning_rate
    this.current_learning_rate = learning_rate
    this.decay = decay
    this.iterations = 0
    this.epsilon = epsilon
    this.beta_1 = beta_1
    this.beta_2 = beta_2
  }
  pre_update_params() {
    if (this.decay) {
      this.current_learning_rate = this.learning_rate * (1 / (1 + this.decay * this.iterations))
    }
  }
  update_params(layer) {
    if (!layer.hasOwnProperty('weight_cache')) {
      layer.weight_momentums = zeros(layer.weights.length, layer.weights[0].length)
      layer.weight_cache = zeros(layer.weights.length, layer.weights[0].length)

      layer.bias_momentums = zeros(layer.biases.length, layer.biases[0].length)
      layer.bias_cache = zeros(layer.biases.length, layer.biases[0].length)
    }
    layer.weight_momentums = add(mult(this.beta_1, layer.weight_momentums), mult(1 - this.beta_1, layer.dweights))
    layer.bias_momentums = add(mult(this.beta_1, layer.bias_momentums), mult(1 - this.beta_1, layer.dbiases))

    let weight_momentums_corrected = div(layer.weight_momentums, 1 - Math.pow(this.beta_1, this.iterations + 1))
    let bias_momentums_corrected = div(layer.bias_momentums, 1 - Math.pow(this.beta_1, this.iterations + 1))
    layer.weight_cache = add(mult(this.beta_2, layer.weight_cache), mult(1 - this.beta_2, pow(layer.dweights, 2)))
    layer.bias_cache = add(mult(this.beta_2, layer.bias_cache), mult(1 - this.beta_2, pow(layer.dbiases, 2)))

    let weight_cache_corrected = div(layer.weight_cache, 1 - Math.pow(this.beta_2, (this.iterations + 1)))
    let bias_cache_corrected = div(layer.bias_cache, 1 - Math.pow(this.beta_2, (this.iterations + 1)))

    layer.weights = add(layer.weights, div(mult(-this.current_learning_rate, weight_momentums_corrected), add(sqrt(weight_cache_corrected), this.epsilon)))
    
    layer.biases = add(layer.biases, div(mult(-this.current_learning_rate, bias_momentums_corrected), add(sqrt(bias_cache_corrected), this.epsilon)))
  }
  post_update_params() {
    this.iterations += 1
  }
}  //Adam optimizer
class Accuracy {
  calculate(predictions, y) {
    let comparisons = this.compare(predictions, y)
    let accuracy = mean(comparisons)
    this.accumulated_sum += sum([comparisons])
    this.accumulated_count += comparisons.length
    return accuracy
  }
  calculate_accumulated(){         
  let accuracy = this.accumulated_sum / this.accumulated_count          
  return accuracy
  }
  new_pass() {
    this.accumulated_sum = 0
    this.accumulated_count = 0
  }
}
class Accuracy_Regression extends Accuracy {
  constructor() {
    super()
    this.name = "Accuracy_Regression"
    this.precision = "None"
  }
  init(y, reinit = false) {
    if (this.precision == "None" || reinit) {
      this.precision = std(y) / 10
    }
  }
  compare(predictions, y) {
    return check(abs(sub(predictions, y)), this.precision, (a, b) => a[0] < this.precision?1:0)
  }
}
class Accuracy_Categorical extends Accuracy {
  constructor() {
    super()
    this.name = "Accuracy_Categorical"
  }
  init(){
  }
  compare(predictions, y) {
    if(shape(y).length == 2){
      y = argmax(y)
    }
   return check(predictions, y, (a, b) => a == b ?1:0)
  }

}