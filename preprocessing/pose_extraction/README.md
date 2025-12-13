Run the command
```
python extract_pose_phoenix.py
```
the pose features will be extracted and placed within this folder as 
```
|_phoenix_pose_features_dev
|_phoenix_pose_features_train
|_phoenix_pose_features_test
```

After that run 
```
python pose_embeddings_MLP.py
```

which will generate three more folders
```
|_phoenix_pose_embeddings_dev
|_phoenix_pose_embeddings_train
|_phoenix_pose_embeddings_test
```
