
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf087-short-circuit-nested/cf087-short-circuit-nested_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z25test_nested_short_circuitiii>:
100000360: 7100001f    	cmp	w0, #0x0
100000364: 7a40c824    	ccmp	w1, #0x0, #0x4, gt
100000368: 7a40d840    	ccmp	w2, #0x0, #0x0, le
10000036c: 1a9fd7e0    	cset	w0, gt
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800020    	mov	w0, #0x1                ; =1
100000378: d65f03c0    	ret
