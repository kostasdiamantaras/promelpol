(ns image-gen.dataset
  (:refer-clojure :exclude [update])
  (:use
   [image-gen.db-info]
   [korma.core]))

(defmacro load-shops
  []
  `(do
     ~@(for [i (range 60)]
         `(defentity ~(symbol (str "shop" i))))))
(load-shops)

(defentity datasales)

(def shop-list (map #(symbol (str "shop" %)) (range 60)))

(defn items-sold-at
  [item shop year month day]
  (-> (select datasales
              (where {:shop_id shop
                      :item_id item
                      :y year
                      :m month
                      :d day})
              (aggregate (sum :item_cnt_day) :sum))
      first
      :sum
      int))

(defn sales-of
  [shop item]
  (select shop
          (where {:item_id item})))

(defn items-of
  [shop]
  (map :item_id (select shop
                        (fields :item_id)
                        (group :item_id))))

;(items-sold (nth shop-list 0))

;(sales-of (nth shop-list 0) 1)


