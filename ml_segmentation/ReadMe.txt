## DATA:
Eerst data_distribution runnen, path aanpassen
Dan data_maken runnen, ook path aanpassen

## Trainen:
Run train_new, in train_utilities_new staan de rest van de functies. 	-- Pas wel je path aan
									-- Het beste is een 'if name ==' toevoegen als je hem zelf wilt runnen :)
									-- Model naam aanpassen naar wat je wilt/iets logisch
									-- Feature naam (data_names) aanpassen naar iets logisch
									-- Als je al features hebt gemaakt dan use_saved = TRUE, anders FALSE
									-- Als je features op wilt slaan met de data_names dan save = TRUE (is al default).

Train_utilities_new: vooral naar main kijken helemaal onderaan, hier worden de rest van de functies aangeroepen.

## Masks krijgen:	
Run create_segmentations, in create_segmentations_util staan de functies, ook hier vooral bij main kijken.
									-- PATH AANPASSEN met een if statement toevoegen
									-- Model naam aanpassen naar hetzelfde als bij train gebruikt