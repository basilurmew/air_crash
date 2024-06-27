from flask import Flask, render_template_string, request
import folium
from table_to_map import map_gen
import pandas as pd
import  numpy as np
# create a flask application
app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == "GET":
        """Create a map object"""

        # тут код индуса идет, его не трогать!!!

        df = pd.read_csv('newCrash.csv')

        df1 = df[np.isnan(df["latitude"]) != True]

        mapObj = map_gen(df1)

        # render the map object
        mapObj.get_root().render()

        # derive the script and style tags to be rendered in HTML head
        header = mapObj.get_root().header.render()

        # derive the div container to be rendered in the HTML body
        body_html = mapObj.get_root().html.render()

        # derive the JavaScript to be rendered in the HTML body
        script = mapObj.get_root().script.render()

        # return a web page with folium map components embeded in it. You can also use render_template.
        return render_template_string(
            """
                <!DOCTYPE html>
                <html>
                    <head>
                        {{ header|safe }}
                        <style>
                            body{
                                display:flex;
                                flex-direction:column;
                            }
                            
                            header{
                                width:100%;
                                height:15%;
                                background-color:black;
                                flex-direction: row;
                                justify-content: center;
                                align-items:center;
                            }
                            
                            h1{
                                color:white;
                            }
                        </style>
                    </head>
                    <body>
                        <header>
                            <h1> B-17 flying home </h1>    
                        </header>
                        
                        {{ body_html|safe }}
                        <form method = "post">
                            <input type = "text" name = "min_year">
                            <input type = "text" name = "max_year">
                        </form>
                        <script>
                            {{ script|safe }}
                        </script>
                    </body>
                </html>
            """,
            header=header,
            body_html=body_html,
            script=script,
        )
    #else:
     #   min_year = request.form.get('min_year')
      #  max_year = request.form.get('max_year')





if __name__ == "__main__":
    app.run()