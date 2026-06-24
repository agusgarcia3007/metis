package hands

import "testing"

func TestCalc(t *testing.T) {
	for _, c := range []struct{ e, w string }{{"84937*2261", "192042557"}, {"(5+3)/2", "4"}, {"2^10", "1024"}, {"10%3", "1"}, {"-3+5", "2"}} {
		g, err := Calc(c.e)
		if err != nil || g != c.w {
			t.Errorf("Calc(%q)=%q,%v want %q", c.e, g, err, c.w)
		}
	}
}
