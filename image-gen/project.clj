(defproject image_gen "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [korma "0.4.3"]
                 [org.clojure/java.jdbc "0.6.1"]
                 [mysql/mysql-connector-java "8.0.16"]]

  :main ^:skip-aot image-gen.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
