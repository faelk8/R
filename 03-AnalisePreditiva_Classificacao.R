setwd("C:/Users/Rafael/Documents/01-Portifolio-R")

# Análise de Crédito
# Um banco quer saber se um cliente pode ou não receber um emprestimo.
# O modelo deve ter acurácia > 75%

# Carga de dados
clientes <- read.csv("dados/clientes.csv", header = TRUE, sep = ",")
str(clientes)

# Função para transformar os dados para o tipo fator
fator <- function(data, variavel){
  for (v in variavel){
    data[[v]] <- as.factor(data[[v]])
  }
  return(data)
}

# Função para Normalização
escala <- function(data, variavel){
  for (v in variavel){
    data[[v]] <- scale(data[[v]], center=T, scale=T)
  }
  return(data)
}


# Normalizando as variáveis
normal <- c("credit.duration.months", "age", "credit.amount")
clientes <- escala(clientes, normal)

# Coletando as variáveis que serão transformadas em Fator
categoria <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
               'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
               'marital.status', 'guarantor', 'residence.duration', 'current.assets',
               'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
               'dependents', 'telephone', 'foreign.worker')


# Convertendo as demais variáveis para fator
clientes <- fator(data=clientes, variavel=categoria)
head(clientes)
str(clientes)

# Divisão em dados de treino e de teste em uma taxa de 70:30 
indexes <- sample(1:nrow(clientes), size = 0.7 * nrow(clientes))
train.data <- clientes[indexes,]
test.data <- clientes[-indexes,]


#####-------------- Feature Selection -----------------
library(caret)  
library(randomForest) 

# Função para Seleção de variáveis
variaveis <- function(dobra=20, variaveis, classe){
  tamanho <- 1:10
  control <- rfeControl(functions = rfFuncs, 
                        method = "cv", 
                        verbose = FALSE, 
                        returnResamp = "all", 
                        number = dobra)
  resultado <- rfe(x = variaveis, 
                     y = classe, 
                     sizes = tamanho, 
                     rfeControl = control)
  return(resultado)
}

# Executando a Função
resultado <- variaveis(variaveis = train.data[,-1],  classe = train.data[,1])

# Resultado
resultado


#####-------------- Árvores de Decisão -----------------
library(rpart)
library(rpart.plot) 
library(ROCR) 
library(e1071)

# Separando atributos e variáveis preditoras
x_teste <- test.data[,-1]
y_teste <- test.data[,1]

# Construindo o modelo inicial com os dados de treino
modelo <- "credit.rating ~ ."
modelo <- as.formula(modelo)
modelo <- rpart(formula = modelo, 
                  method = "class", 
                  data = train.data, 
                  control = rpart.control(minsplit = 20, cp = 0.05))

# Prevendo e avaliando o resultado
previsao <- predict(modelo, x_teste, type = "class")
confusionMatrix(data = previsao, reference = y_teste, positive = "1") 


# Seleção de variáveis
modelo <- "credit.rating ~ ."
modelo <- as.formula(modelo)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
modelo <- train(modelo, data = train.data, method = "rpart", trControl = control)
importancia <- varImp(modelo, scale = FALSE)
plot(importancia, cex.lab = 0.5)

## Construindo um modelo com as variáveis selecionadas
formula <- "credit.rating ~ account.balance + credit.amount + credit.duration.months + previous.credit.payment.status"
formula <- as.formula(formula)
modelo <- rpart(formula = formula, 
                   method = "class",
                   data = train.data, 
                   control = rpart.control(minsplit = 20, cp = 0.05),
                   parms = list(prior = c(0.7, 0.3)))

# Previsões e Avaliação do Resultado
previsao <- predict(modelo, x_teste, type = "class")
confusionMatrix(data = previsao, reference = y_teste, positive = "1") 

# Plot do Modelo com Curva ROC
curvaROC <- function(previsao, title.text){
  perf <- performance(previsao, "tpr", "fpr")
  plot(perf,col = "black",lty=1, lwd=2,
       main=title.text, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
  abline(0,1, col="red")
  auc <- performance(previsao,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
  
}

previsao <- predict(modelo, x_teste, type="prob")
previsao <- previsao[,2]
previsto <- prediction(previsao, y_teste)
par(mfrow = c(1,2))
curvaROC(previsto, title.text = "Curva ROC - Árvore de Decisão")



#####-------------- Regressão Logística -----------------

# Construindo o modelo de regressão
modelo_rl <- "credit.rating ~ ."
modelo_rl <- as.formula(modelo_rl)
modelo_rl <- glm(formula = modelo_rl, data = train.data, family = "binomial")

# Visualizando o resultado
summary(modelo_rl)

# Previsão e avaliação
previsao <- predict(modelo_rl, x_teste, type = "response")
previsao <- round(previsao)
confusionMatrix(data = previsao, reference = y_teste, positive = '1')

# Seleção de variáveis método glm
modelo <- "credit.rating ~ ."
modelo <- as.formula(modelo)
control <- trainControl(method="repeatedcv", number = 10, repeats = 2)
model <- train(modelo, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance)

# Construindo o modelo com variáveis selecionadas
modelo <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
modelo <- as.formula(modelo)
modelo <- glm(formula = modelo, data = train.data, family = "binomial")

# Visualizando o modelo
summary(modelo)

# Previsões e avaliações
previsao <- predict(modelo, x_teste, type = "response") 
previsao <- round(previsao)
confusionMatrix(data = previsao, reference = y_teste, positive = '1')

previsao <- predict(modelo, x_teste, type="response")
previsto <- prediction(previsao, y_teste)
par(mfrow=c(1,2))
curvaROC(previsto, title.text = "Curva ROC - Regressão Logística") 




#####-------------- Redes Neurais -----------------


# Transformação de dados
treino <- train.data
teste <- test.data

for (c in categoria){
  n_train <- make.names(train.data[[c]])
  treino[[c]] <- n_train
  n_teste <- make.names(test.data[[c]])
  teste[[c]] <- n_teste
}

treino <- fator(data=treino, variavel=categoria)
teste <- fator(data=teste, variavel=categoria)
x_teste <- teste[,-1]
y_teste <- teste[,1]

# Construindo o modelo com dados de treino
modelo <- "credit.rating ~ ."
modelo <- as.formula(modelo)
modelo <- train(modelo, data = treino, method = "nnet")

# Visualizando resultados do modelo
print(modelo)

# Previsões e avaliação
previsao <- predict(modelo, x_teste, type = "raw")
confusionMatrix(data = previsao, reference = y_teste, positive = "X1") 

# Seleção de variáveis
modelo <- "credit.rating ~ ."
modelo <- as.formula(modelo)
control <- trainControl(method="repeatedcv", number = 10, repeats = 2)
modelo <- train(modelo, data = treino, method = "nnet", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance, cex.lab = 0.5)

# Construindo o modelo com as variáveis selecionadas
modelo <- "credit.rating ~ account.balance + credit.purpose + savings + current.assets + foreign.worker + previous.credit.payment.status"
modelo <- as.formula(modelo)
modelo <- train(modelo, data = treino, method = "nnet")

# Previsões e avaliação
previsao <- predict(previsao, x_teste, type = "raw")
confusionMatrix(data = previsao, reference = y_teste, positive = "X1") 

# Visualização de Hiperparâmetros

previsao <- predict(modelo, x_teste, type="prob")
previsao <- previsao[,2]
previsto <- prediction(previsao, y_teste)
par(mfrow=c(1,2))
curvaROC(previsto, title.text="Curva ROC - Redes Neurais")



