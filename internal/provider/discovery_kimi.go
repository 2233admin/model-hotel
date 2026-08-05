package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/hugalafutro/model-hotel/internal/debuglog"
	"github.com/hugalafutro/model-hotel/internal/model"
	"github.com/hugalafutro/model-hotel/internal/util"
)

// KimiModelsResponse is the response from GET /v1/models on api.kimi.com/coding
type KimiModelsResponse struct {
	Data     []KimiModel `json:"data"`
	Object  string `json:"object"`
	FirstID string `json:"first_id"`
	LastID  string `json:"last_id"`
	HasMore bool   `json:"has_more"`
}

type KimiModel struct {
	ID                 string `json:"id"`
	Created           int64  `json:"created"`
	CreatedAt         string `json:"created_at"`
	Object            string `json:"object"`
	DisplayName       string `json:"display_name"`
	Type             string `json:"type"`
	ContextLength    int    `json:"context_length"`
	SupportsReasoning bool   `json:"supports_reasoning"`
	SupportsImageIn  bool   `json:"supports_image_in"`
	SupportsVideoIn  bool   `json:"supports_video_in"`
	SupportsThinking string `json:"supports_thinking_type"` // "only", "auto", "never"
}

func (d *DiscoveryService) discoverKimi(ctx context.Context, provider *Provider, apiKey string) ([]*model.Model, error) {
	baseURL := util.SanitizeBaseURL(provider.BaseURL)

	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+apiKey)
	headers.Set("Content-Type", "application/json")

	bodyBytes, err := d.fetchURL(ctx, "GET", baseURL+"/v1/models", headers)
	if err != nil {
		debuglog.Error("discovery: kimi fetch models failed", "provider", provider.Name, "provider_id", provider.ID, "error", err)
		return nil, fmt.Errorf("kimi: failed to fetch models for provider %s: %w", provider.Name, err)
	}

	var kimiResp KimiModelsResponse
	if err := json.Unmarshal(bodyBytes, &kimiResp); err != nil {
		debuglog.Error("discovery: kimi json decode failed", "provider", provider.Name, "provider_id", provider.ID, "error", err)
		return nil, fmt.Errorf("kimi: failed to decode response for provider %s: %w", provider.Name, err)
	}

	live := make([]*model.Model, 0, len(kimiResp.Data))
	for _, m := range kimiResp.Data {
		// video_in implies image_in; must be valid JSON array for JSONB column
		var inputMod string
		if m.SupportsVideoIn {
			inputMod = `["image","video"]`
		} else if m.SupportsImageIn {
			inputMod = `["image"]`
		} else {
			inputMod = `[]`
		}

		// Reasoning: prefer the dedicated bool flag, fall back to thinking_type
		reasoning := m.SupportsReasoning
		if !reasoning && (m.SupportsThinking == "only" || m.SupportsThinking == "auto") {
			reasoning = true
		}
		capabilities := model.Capability{Streaming: true, Reasoning: reasoning}
		capJSON, _ := json.Marshal(capabilities)

		ctxLen := int(m.ContextLength)
		live = append(live, &model.Model{
			ID:           uuid.New(),
			ProviderID:   provider.ID,
			ModelID:      m.ID,
			Name:         m.ID,
			DisplayName:  m.DisplayName,
			Capabilities: string(capJSON),
			Params:       "{}",
			Modality:        "text", // Coding Plan is text-focused
			InputModalities:  inputMod,
			OutputModalities: "[]",
			ContextLength: &ctxLen,
			OwnedBy:      "moonshot",
			CreatedAt:    parseKimiTime(m.CreatedAt),
		})
	}

	if len(live) == 0 {
		debuglog.Warn("discovery: kimi /v1/models returned no models, skipping", "provider", provider.Name, "provider_id", provider.ID)
		return live, nil
	}

	debuglog.Info("discovery: kimi discovered models", "provider", provider.Name, "provider_id", provider.ID, "count", len(live))
	return live, nil
}

// parseKimiTime parses Kimi's ISO8601-like timestamp string into time.Time.
// Kimi returns formats like "2025-04-30T19:01:00Z" or "2025-04-30T19:01:00+08:00".
func parseKimiTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		// Fallback: try parsing without timezone offset
		t, err = time.Parse("2006-01-02T15:04:05", s)
		if err != nil {
			debuglog.Warn("discovery: kimi: failed to parse timestamp", "value", s)
			return time.Time{}
		}
	}
	return t
}
