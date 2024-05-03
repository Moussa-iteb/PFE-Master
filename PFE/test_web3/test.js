const Web3 = require('web3');
const fs = require('fs');

const trufflePath = 'C:/Users/ACER/PFE/test_web3/';
const truffleFile = JSON.parse(fs.readFileSync(trufflePath + 'earn_project.json'));
const abi = truffleFile.abi;
const bytecode = truffleFile.bytecode;
const web3 = new Web3('http://localhost:9545');

const contractAddress = ' 0x944Aef8d5fac7fa0C07FE70dCE1f07DF823A46f9';
const contract = new web3.eth.Contract(abi, contractAddress);

const privateKey = ' 0x944Aef8d5fac7fa0C07FE70dCE1f07DF823A46f9';
const account = web3.eth.accounts.privateKeyToAccount(privateKey);
web3.eth.defaultAccount = account.address;

web3.eth.getBalance(account.address)
  .then(balance => {
    console.log('Account balance:', web3.utils.fromWei(balance, 'ether'));
    console.log('Default account:', account.address);
    console.log('Is connected:', web3.eth.net.isListening());
  
    contract.methods.getDataByDate(123).call()
      .then(valuetx => {
        console.log(valuetx);
      })
      .catch(err => {
        console.error('Error calling contract method:', err);
      });
  })
  .catch(err => {
    console.error('Error getting account balance:', err);
  });
