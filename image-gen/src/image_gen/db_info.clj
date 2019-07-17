(ns image-gen.db-info
  (:use [korma.db]
        [korma.core]))

(defdb exam-db (mysql {:db "sales"
                       :user "redacted"
                       :password "redacted"}))

(defentity data)
