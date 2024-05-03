const earn_project= artifacts.require("earn_project");

module.exports = function (deployer) {
  deployer.deploy(earn_project);
};
