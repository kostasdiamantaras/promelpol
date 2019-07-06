(ns image-gen.core
  (:refer-clojure :exclude [update])
  (:use
   [image-gen.db-info]
   [korma.core]
   [image-gen.dataset]))

(defn create-directories
  [shops]
  (.mkdir (java.io.File. "./imageset/"))
  (doseq [shop shops]
    (.mkdir (java.io.File. (str "./imageset/" (str shop))))
    (.mkdir (java.io.File. (str "./imageset/" (str shop) "/train")))
    (.mkdir (java.io.File. (str "./imageset/" (str shop) "/test")))))

;(create-directories shop-list)

(defn matrix
  [width height]
  (mapv (fn [_] (mapv (constantly 0) (range width))) (range height)))

(defn calendar
  [years width height]
  (mapv (fn [_] (matrix width height)) (range years)))

(defn csv
  [matrix]
  (reduce str (flatten (interpose "\n" (pmap #(interpose "," %) matrix)))))

(defn put
  [cal year month day value]
  (let [year (if (= month 12)
               (- year 2012)
               (- year 2013))
        month (if (= month 12) 0 month)]
    (update-in cal [year month day] (fn [_ i] i) value)))

(defn sales-calendar
  [shop item]
  (print "sales calendar")
  (let [calendar (calendar 3 31 12)]
    (reduce #(put %1
                  (:y %2)
                  (:m %2)
                  (:d %2)
                  (int (:amount %2)))
            calendar
            (sales-of 'shop0 item))))

(defn write!
  [shop item year [train test]]
  (spit (str "./imageset/" (str shop) "/train/" item "_" year ".csv") (csv train))
  (spit (str "./imageset/" (str shop) "/test/" item "_" year ".csv") (csv test))
  true)

(defn split
  [annual-sales]
  [(butlast annual-sales) (last annual-sales)])

(defn save
  [shop item calendar]
  (doseq [c (map list calendar [2013 2014 2015])]
    (print "\n im i \n")
    (write! shop item (second c) (split (first c)))))

(defn a
  [shop]
  (let [items (items-of shop)]
    (print "Processing " (str shop) "\n")
    (pmap #(-> %
               (sales-calendar shop)
               (save shop %))
          items)))

(defn -main
  [& args]
  )
