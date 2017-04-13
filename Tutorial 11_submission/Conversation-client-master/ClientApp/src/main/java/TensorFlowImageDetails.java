import org.json.JSONArray;
import org.json.JSONObject;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Created by Naga on 22-03-2017.
 */
@WebServlet(name = "TensorFlowImageDetails", urlPatterns = "/tensorFlowImageDetails")
public class TensorFlowImageDetails extends HttpServlet {

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        StringBuilder buffer = new StringBuilder();
        BufferedReader reader = req.getReader();
        String line;
        while ((line = reader.readLine()) != null) {
            buffer.append(line);
        }
        String data = buffer.toString();
        System.out.println(data);
        String output = "";
        JSONObject params = new JSONObject(data);
        JSONObject result = params.getJSONObject("result");
        JSONObject parameters = result.getJSONObject("parameters");
        if (parameters.get("null").toString().equals("clear")){
            Data data_ob = Data.getInstance();
            JSONObject js1 = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put(" ");
            js1.put("data", jsonArray);
            data_ob.setData(js1.toString());
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "screen is cleared");
            js.put("displayText", "screen is cleared");
            js.put("source", "image database");
            output = js.toString();
        }
        else{
            String classs = parameters.getString("instruments").toString();
            JSONObject js1 = new JSONObject();

            JSONArray jsonArray = new JSONArray();
            jsonArray.put("http://i63.tinypic.com/10hq0wx.jpg");
            jsonArray.put("http://i66.tinypic.com/2usx6xw.jpg");
            jsonArray.put("http://i67.tinypic.com/1zmdoph.jpg");
//            jsonArray.put("http://i63.tinypic.com/5xju40.jpg");
//            jsonArray.put("http://i64.tinypic.com/2llf09e.jpg");
            jsonArray.put("http://i66.tinypic.com/invhb7.jpg");
            jsonArray.put("http://i64.tinypic.com/v3p1dl.jpg");
            jsonArray.put("http://i64.tinypic.com/1fg5yf.jpg");
//            jsonArray.put("http://i64.tinypic.com/v2ur6w.jpg");
//            jsonArray.put("http://i66.tinypic.com/2gw5091.jpg");
//            jsonArray.put("http://i65.tinypic.com/5bpf6a.jpg");
//            jsonArray.put("http://i64.tinypic.com/rsud7o.jpg");
//            jsonArray.put("http://i63.tinypic.com/2cpe149.jpg");
//            jsonArray.put("http://i64.tinypic.com/20z33vq.jpg");
//            jsonArray.put("http://i65.tinypic.com/11kb76g.jpg");
            js1.put("imageBase64", jsonArray);

            String url = "http://ec2-34-200-227-114.compute-1.amazonaws.com/";
//            String url = "https://mighty-headland-54930.herokuapp.com";
//            String url = "https://gentle-ocean-30903.herokuapp.com/api/predict";


//        String url = "https://image-details.herokuapp.com/imageDetails";
            URL obj = new URL(url);
            HttpURLConnection con = (HttpURLConnection) obj.openConnection();

            //add reuqest header
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            con.setRequestProperty("Accept-Language", "en-US,en;q=0.5");
            String urlParameters = "sn=C02G8416DRJM&cn=&locale=&caller=&num=12345";

            // Send post request
            con.setDoOutput(true);
            DataOutputStream wr = new DataOutputStream(con.getOutputStream());
//        wr.writeBytes(urlParameters);
            wr.writeBytes(js1.toString());
            wr.flush();
            wr.close();

            int responseCode = con.getResponseCode();

            System.out.println("\nSending 'POST' request to URL : " + url);
            System.out.println("Post parameters : " + urlParameters);
            System.out.println("Response Code : " + responseCode);

            BufferedReader in = new BufferedReader(
                    new InputStreamReader(con.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();

            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();

            String[] c = response.toString().replace("[","").replace("]","").replace("'","").split(",");
            Data data_ob = Data.getInstance();
            JSONObject newData=new JSONObject();
            JSONArray js = new JSONArray();
            for (int i=0; i<c.length; i++){
                if (c[i].trim().equals(classs))
                    js.put(jsonArray.get(i));
            }
            newData.put("data", js);
            data_ob.setData(newData.toString());
            //print result
//            System.out.println(response.toString());
            JSONObject js2 = new JSONObject();
            js2.put("speech", "Images are displayed on the screen");
            js2.put("displayText", "Images are displayed on the screen");
            js2.put("source", "database");
            output = js2.toString();
        }

        resp.setHeader("Content-type", "application/json");
        resp.getWriter().write(output);
    }
}
