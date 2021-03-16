pipeline{
    agent any
        stages{
        stage('checkout'){
            steps{
                git branch: 'main', credentialsId: '7e4950e0-0efe-45b9-bfc2-cccdef976cf9', url: 'https://github.com/AnalyticsSolution/IncidentPrediction.git'
            }
        }
        stage('train'){
            steps{
                bat label: '', script: '''cd Scripts/
python training_v1.py '''
            }
        }
        stage('predict'){
            steps{
                bat label: '', script: '''cd Scripts/
python Prediction_v1.py '''
            }
		}
	}
	}
