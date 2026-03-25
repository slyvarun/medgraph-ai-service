package com.medgraph.nexus_gateway;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api")
public class AIController {

    private final String PYTHON_AI_URL = "https://medgraph-ai-service.onrender.com/ask";

    @PostMapping("/query")
    public String askAI(@RequestBody String userQuestion) {
        RestTemplate restTemplate = new RestTemplate();
        
        // 1. Prepare the request for Python (FastAPI expects a string body)
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.TEXT_PLAIN);
        HttpEntity<String> request = new HttpEntity<>(userQuestion, headers);

        // 2. Call the Python Service
        String response = restTemplate.postForObject(PYTHON_AI_URL, request, String.class);
        
        return "Nexus AI System Response: " + response;
    }
}
