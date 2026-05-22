# Этап 1: Сборка JAR-файла с помощью sbt (этот этап работает)
FROM hseeberger/scala-sbt:17.0.2_1.6.2_2.12.15 AS builder
WORKDIR /app
COPY . .
RUN sbt assembly

#  с использованием ОФ образа Spark
FROM apache/spark:3.5.1
USER root
WORKDIR /opt/spark

# собранный jar-архив
COPY --from=builder /app/target/scala-2.12/*.jar ./datamart.jar

#  драйвер JDBC для Oracle
RUN curl -O https://repo1.maven.org/maven2/com/oracle/database/jdbc/ojdbc8/21.9.0.0/ojdbc8-21.9.0.0.jar

# Путь к spark-submit в официальном образе находится в /opt/spark/bin/
CMD ["/opt/spark/bin/spark-submit", \
     "--class", "ru.itmo.bigdata.DataMartApp", \
     "--jars", "ojdbc8-21.9.0.0.jar", \
     "datamart.jar"]