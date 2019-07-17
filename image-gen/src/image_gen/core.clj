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
    (.mkdir (java.io.File. (str "./imageset/" (str shop) "/input")))
    (.mkdir (java.io.File. (str "./imageset/" (str shop) "/target")))))

(defn matrix
  [width height]
  (mapv (fn [_] (mapv (constantly 0) (range width))) (range height)))

(defn calendar
  [years width height]
  (mapv (fn [_] (matrix width height)) (range years)))

(defn csv
  "Turns a 2d matrix to a csv (only works for non-string data)"
  [matrix]
  (reduce str (flatten (interpose "\n" (pmap #(interpose "," %) matrix)))))

(defn put
  "Updates the specified date in the calendar with the correct number of sales"
  [cal year month day value]
  (let [year (if (= month 12)
               (- year 2012)
               (- year 2013))
        month (if (= month 12) 0 month)]
    (update-in cal [year month day] (fn [_ i] i) value)))

(defn sales-calendar
  "Create a 3d matrix that holds the sales data"
  [shop item-sales]
  (let [calendar (calendar 3 31 12)]
    (reduce #(put %1
                  (:y %2)
                  (:m %2)
                  (dec (:d %2))
                  (int (:amount %2)))
            calendar
            item-sales)))

(defn write!
  "Turns the data into csv and saves them in the appropriate folders"
  [shop item year [train test]]
  (spit (str "./imageset/shop" shop "/input/" item "_" year ".csv") (csv train))
  (spit (str "./imageset/shop" shop "/target/" item "_" year ".csv") (csv [test]))
  true)

(defn split
  "Splits the data into input and target.
  Input data is 0-10 (December to October) and target is 11 (or November)"
  [annual-sales]
  [(butlast annual-sales) (last annual-sales)])

(defn save
  "Seperates sales per year and writes them to disk"
  [shop item calendar]
    (for [c (map list calendar [2013 2014 2015])]
      (write! shop item (second c) (split (first c)))))

(defn process
  [shop]
  (let [items (items-of shop)]
    (pmap (fn [item]
            ((comp #(save shop item %)
                   #(sales-calendar shop %)
                   #(sales-of shop %)) item))
          items)))

(defmacro process-all
  [max]
  `(do
     ~@(doseq [i (range 60)]
         `(defentity ~(symbol (str "shop" i))))))

(defn -main
  [& args]
  (create-directories shop-list)
  (pmap process (range 60)))

