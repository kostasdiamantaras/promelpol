(ns image-gen.dataset
  (:refer-clojure :exclude [update])
  (:use
   [image-gen.db-info]
   [korma.core]))

(defmacro load-shops
  []
  `(do
     ~@(doseq [i (range 60)]
         `(defentity ~(symbol (str "shop" i))))))
(load-shops)

(defentity datasales)

(def shop-list (map #(symbol (str "shop" %)) (range 60)))

(defn sales-of
  [shop item]
  (select (nth shop-list shop)
          (fields :y :m :d :amount)
          (where {:item_id item})))

(defn items-of
  [shop]
  (map :item_id
       (select (nth shop-list shop)
               (fields :item_id)
               (group :item_id))))

