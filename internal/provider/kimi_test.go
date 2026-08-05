package provider

import (
    "encoding/json"
    "fmt"
    "testing"
    "github.com/hugalafutro/model-hotel/internal/model"
)

func TestKimiCapabilityJSON(t *testing.T) {
    caps := model.Capability{Streaming: true, Reasoning: true}
    data, _ := json.Marshal(caps)
    fmt.Printf("Capabilities: %s\n", string(data))
    
    caps2 := model.Capability{Streaming: true}
    data2, _ := json.Marshal(caps2)
    fmt.Printf("Capabilities (streaming only): %s\n", string(data2))
}
