#!/usr/bin/perl
#
#
use File::Basename;

$spkfn = $ARGV[0];
open F, $spkfn;
while(<F>){
  ($a, $b) = split;
  $H{$a} = $b;
  #print "$a -> $b\n";
}
close F;

while(<STDIN>){
  chomp; $line = $_;
  @a = split /\|/;
  $base = basename($a[0], ".vec192");
  $spk = substr $base, 0, 7;
  die $a[0], $spk if(!exists $H{$spk});
  print "$line|$H{$spk}\n";
}




