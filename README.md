# 1st place solution
Thanks to the organizers for the interesting competition, we were able to reach a rapid growth in the field of hash retrieval. One of them, 48-bit hash code, we think is a very big challenge for us, it brought us some trouble, but we gained a lot of growth, and we hope this competition can be run all the time.

# Summary
Below I will give a brief explanation of our program.
1.We use the backbone network  to train for the classification task and obtain a robust baseline.

2.After that, we perform feature extraction on the query and gallery images.

Feature enhancement is performed separately for the trained model.We use the pyretri repository, feature enhancement is performed on the features that have been extracted.

3.Fusion of the above trained models.

4.n this step the queries and galleries are grouped according to the similarity matrix, and the same set of queries is a class.

5.Use the MD5 encryption method of hashlib to generate a 12-bit hexadecimal code for the image, and then convert it to a 48-bit hashcode.


# Conclusion
In summary, our overall process is to extract features with a powerful model, then perform feature enhancement, perform model fusion, group query and gallery according to features, and finally assign hash codes. Thanks again for hosting this competition, which enriched our intellectual perspective.

# Reference
[1] Fang Y, Wang W, Xie B, et al. Eva: Exploring the limits of masked visual representation learning at scale[J]. arXiv preprint arXiv:2211.07636, 2022.

[2] Bao H, Dong L, Piao S, et al. Beit: Bert pre-training of image transformers[J]. arXiv preprint arXiv:2106.08254, 2021.
