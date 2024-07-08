from src import model
import pandas as pd
annotations = '/blue/ewhite/everglades/Zooniverse/cleaned_test/test_resized_no_nan.csv'
df = pd.read_csv(annotations)
m = model.extract_backbone(path="/blue/ewhite/everglades/Zooniverse//20230426_082517/species_model.pl", annotations=df)
m.create_trainer()
m.trainer.strategy.connect(m)
m.trainer.save_checkpoint("/blue/ewhite/everglades/Zooniverse/20230426_082517/species_model_backbone.pl")