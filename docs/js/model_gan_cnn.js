

const model = await  tf.loadModel('docs/tfjs/gan-cnn/model.json');
//const cnn = await tf.loadModel('../docs/tfjs/cnn/model.json');

if(model == NaN ){
    prompt("Error load the model ")
}
