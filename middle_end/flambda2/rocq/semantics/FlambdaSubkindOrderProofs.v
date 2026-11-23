(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaSubkindOrderSpec.v. There is no
   statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaSubkindSpec
  FlambdaSubkindOrderSpec.
Import ListNotations.

(* ======================================================================== *)
(* Small facts                                                              *)
(* ======================================================================== *)

Lemma boxable_number_eqb_true (n n' : boxable_number) :
  boxable_number_eqb n n' = true <-> n = n'.
Proof.
  unfold boxable_number_eqb.
  destruct (boxable_number_eq_dec n n') as [Heq | Hne]; split;
    [ intros _; exact Heq | reflexivity | discriminate | contradiction ].
Qed.

Lemma existsb_eqb_In (i : Z) (l : list Z) :
  existsb (Z.eqb i) l = true <-> In i l.
Proof.
  rewrite existsb_exists. split.
  - intros (j & Hin & Heq). apply Z.eqb_eq in Heq. subst j. exact Hin.
  - intros Hin. exists i. split; [ exact Hin | apply Z.eqb_refl ].
Qed.

Lemma consts_among (consts consts' : list Z) (i : Z) :
  forallb (fun i => existsb (Z.eqb i) consts') consts = true ->
  existsb (Z.eqb i) consts = true ->
  existsb (Z.eqb i) consts' = true.
Proof.
  intros Hall Hi. apply existsb_eqb_In in Hi. rewrite forallb_forall in Hall.
  exact (Hall i Hi).
Qed.

(* The test of the shapes reads the first entry for the tag. *)
Lemma block_of_shapes_by_tag (statics : static_env)
    (shapes : constructor_shapes) (tag : Z) (fields : list value) :
  block_of_shapes statics shapes tag fields
  = match shape_of_tag tag shapes with
    | None => false
    | Some None => true
    | Some (Some sks) => fields_of_subkinds statics sks fields
    end.
Proof.
  induction shapes as [ | tag' rest IH | tag' sks rest IH];
    cbn [block_of_shapes shape_of_tag]; [ reflexivity | | ];
    destruct (Z.eqb tag tag'); try reflexivity; exact IH.
Qed.

Lemma value_of_variant (statics : static_env) (consts : list Z)
    (shapes : constructor_shapes) (v : value) :
  value_of_subkind statics (Subkind_variant consts shapes) v
  = match v with
    | Value_const (Tagged_immediate i) =>
      is_tagged_immediate v && existsb (Z.eqb i) consts
    | _ =>
      match block_of statics v with
      | Some (tag, fields) => block_of_shapes statics shapes tag fields
      | None => false
      end
    end.
Proof. reflexivity. Qed.

Ltac by_value v :=
  destruct v; cbv beta iota;
  try (match goal with c : const |- _ => destruct c; cbv beta iota end).

Lemma variant_weaken (statics : static_env) (consts consts' : list Z)
    (shapes shapes' : constructor_shapes) (v : value) :
  (∀ i : Z,
     existsb (Z.eqb i) consts = true -> existsb (Z.eqb i) consts' = true) ->
  (∀ (tag : Z) (fields : list value),
     block_of_shapes statics shapes tag fields = true ->
     block_of_shapes statics shapes' tag fields = true) ->
  value_of_subkind statics (Subkind_variant consts shapes) v = true ->
  value_of_subkind statics (Subkind_variant consts' shapes') v = true.
Proof.
  intros Hconsts Hshapes. rewrite !value_of_variant.
  by_value v;
    first
      [ intros H; apply andb_true_iff in H; destruct H as [Hrange Hi];
        rewrite Hrange; exact (Hconsts _ Hi)
      | destruct (block_of statics _) as [[? ?] | ];
          [ apply Hshapes | exact (fun H => H) ] ].
Qed.

Lemma constants_only_tagged (statics : static_env) (consts : list Z)
    (v : value) :
  value_of_subkind statics (Subkind_variant consts Shapes_nil) v = true ->
  is_tagged_immediate v = true.
Proof.
  rewrite value_of_variant.
  by_value v;
    first
      [ intros H; apply andb_true_iff in H; exact (proj1 H)
      | destruct (block_of statics _) as [[? ?] | ];
          cbn [block_of_shapes]; discriminate ].
Qed.

Lemma anything_says_at_least (b : subkind) :
  subkind_says_at_least Subkind_anything b = true -> b = Subkind_anything.
Proof. destruct b; cbn; intros H; [ reflexivity | discriminate.. ]. Qed.

Lemma says_nothing_weaken (nullable : bool) (sk : subkind) (nullable' : bool)
    (sk' : subkind) :
  negb nullable || nullable' = true ->
  subkind_says_at_least sk sk' = true ->
  says_nothing nullable sk = true ->
  says_nothing nullable' sk' = true.
Proof.
  unfold says_nothing. intros Hnull Hsk H. apply andb_true_iff in H.
  destruct H as [Hn Hany]. destruct sk; try discriminate Hany.
  apply anything_says_at_least in Hsk. subst sk' nullable.
  cbn [negb orb] in Hnull. rewrite Hnull. reflexivity.
Qed.

(* ======================================================================== *)
(* What the test accepts                                                    *)
(* ======================================================================== *)

Fixpoint value_of_subkind_weaken (statics : static_env) (a b : subkind)
    (v : value) {struct a} :
  subkind_says_at_least a b = true ->
  value_of_subkind statics a v = true ->
  value_of_subkind statics b v = true
with block_of_shapes_weaken (statics : static_env) (seen : list Z)
    (shapes shapes' : constructor_shapes) (tag : Z) (fields : list value)
    {struct shapes} :
  shapes_say_at_least seen shapes shapes' = true ->
  existsb (Z.eqb tag) seen = false ->
  block_of_shapes statics shapes tag fields = true ->
  block_of_shapes statics shapes' tag fields = true
with fields_of_subkinds_weaken (statics : static_env)
    (sks sks' : field_subkinds) (fields : list value) {struct sks} :
  fields_say_at_least sks sks' = true ->
  fields_of_subkinds statics sks fields = true ->
  fields_of_subkinds statics sks' fields = true.
Proof.
  - destruct b as [ | | n' | consts' shapes'].
    + intros _ _. reflexivity.
    + destruct a as [ | | n | consts shapes]; cbn [subkind_says_at_least];
        intros Hle Hv; try discriminate Hle.
      * exact Hv.
      * destruct shapes; try discriminate Hle.
        exact (constants_only_tagged statics consts v Hv).
    + destruct a as [ | | n | consts shapes]; cbn [subkind_says_at_least];
        intros Hle Hv; try discriminate Hle.
      apply boxable_number_eqb_true in Hle. subst n'. exact Hv.
    + destruct a as [ | | n | consts shapes]; cbn [subkind_says_at_least];
        intros Hle Hv; try discriminate Hle.
      apply andb_true_iff in Hle. destruct Hle as [Hconsts Hshapes].
      apply (variant_weaken statics consts consts' shapes shapes' v);
        [ intros i; apply consts_among; exact Hconsts
        | intros tag fields;
          exact
            (block_of_shapes_weaken statics [] shapes shapes' tag fields
               Hshapes eq_refl)
        | exact Hv ].
  - destruct shapes as [ | tag0 rest | tag0 sks rest];
      cbn [shapes_say_at_least block_of_shapes]; intros Hle Hseen Hv;
      [ discriminate Hv | | ];
      apply andb_true_iff in Hle; destruct Hle as [Hhead Hrest];
      (destruct (Z.eqb tag tag0) eqn:Htag;
       [ apply Z.eqb_eq in Htag; subst tag0; rewrite Hseen in Hhead;
         cbn [orb] in Hhead; rewrite block_of_shapes_by_tag
       | apply
           (block_of_shapes_weaken statics (tag0 :: seen) rest shapes' tag
              fields Hrest);
         [ cbn [existsb]; rewrite Htag, Hseen; reflexivity | exact Hv ] ]).
    + destruct (shape_of_tag tag shapes') as [[sks' | ] | ];
        try discriminate Hhead.
      reflexivity.
    + destruct (shape_of_tag tag shapes') as [[sks' | ] | ];
        [ exact (fields_of_subkinds_weaken statics sks sks' fields Hhead Hv)
        | reflexivity
        | discriminate Hhead ].
  - destruct sks as [ | nullable sk rest];
      destruct sks' as [ | nullable' sk' rest']; cbn [fields_say_at_least];
      intros Hle Hv; try discriminate Hle.
    + exact Hv.
    + destruct fields as [ | v fields]; cbn [fields_of_subkinds] in Hv |- *;
        [ discriminate Hv | ].
      apply andb_true_iff in Hle. destruct Hle as [Hle Hrest].
      apply andb_true_iff in Hle. destruct Hle as [Hnull Hsk].
      apply andb_true_iff in Hv. destruct Hv as [Hv Hfields].
      apply andb_true_iff in Hv. destruct Hv as [Hkind Hfield].
      rewrite Hkind,
        (fields_of_subkinds_weaken statics rest rest' fields Hrest Hfields),
        andb_true_r.
      cbn [andb]. revert Hfield.
      by_value v; intros Hfield;
        first
          [ exact (says_nothing_weaken _ _ _ _ Hnull Hsk Hfield)
          | rewrite Hfield in Hnull; exact Hnull
          | exact (value_of_subkind_weaken statics sk sk' _ Hsk Hfield) ].
Qed.

Lemma value_at_weaken (statics : static_env) (a b : param_subkind)
    (v : value) :
  param_says_at_least a b = true ->
  value_at statics a v = true ->
  value_at statics b v = true.
Proof.
  destruct a as [na ska], b as [nb skb].
  unfold param_says_at_least, value_at. cbn [fst snd]. intros Hle.
  apply andb_true_iff in Hle. destruct Hle as [Hnull Hsk].
  by_value v; intros Hv;
    first
      [ exact (says_nothing_weaken _ _ _ _ Hnull Hsk Hv)
      | rewrite Hv in Hnull; exact Hnull
      | exact (value_of_subkind_weaken statics ska skb _ Hsk Hv) ].
Qed.

Lemma value_at_says_nothing (statics : static_env) (b : param_subkind)
    (v : value) :
  says_nothing (fst b) (snd b) = true -> value_at statics b v = true.
Proof.
  destruct b as [nb skb]. cbn [fst snd]. unfold says_nothing. intros H.
  apply andb_true_iff in H. destruct H as [Hn Hany].
  destruct skb; try discriminate Hany. subst nb.
  unfold value_at. cbn [fst snd]. by_value v; reflexivity.
Qed.

Theorem values_of_subkinds_weaken : values_of_subkinds_weaken_statement.
Proof.
  intros statics vs a b. revert vs a.
  induction b as [ | pb b IH]; intros vs a Hle Hv.
  - destruct vs; reflexivity.
  - destruct vs as [ | v vs]; [ reflexivity | ]. cbn [values_of_subkinds].
    cbn [params_say_at_least] in Hle. destruct a as [ | pa a].
    + apply andb_true_iff in Hle. destruct Hle as [Hnothing Hrest].
      rewrite (value_at_says_nothing statics pb v Hnothing),
        (IH vs [] Hrest);
        [ reflexivity | ].
      destruct vs; reflexivity.
    + apply andb_true_iff in Hle. destruct Hle as [Hhead Hrest].
      cbn [values_of_subkinds] in Hv. apply andb_true_iff in Hv.
      destruct Hv as [Hv Hvs].
      rewrite (value_at_weaken statics pa pb v Hhead Hv), (IH vs a Hrest Hvs).
      reflexivity.
Qed.

(* ======================================================================== *)
(* Reflexive                                                                *)
(* ======================================================================== *)

Fixpoint subkind_says_at_least_refl (a : subkind) {struct a} :
  subkind_says_at_least a a = true
with shapes_say_at_least_refl (all : constructor_shapes) (seen : list Z)
    (cur : constructor_shapes) {struct cur} :
  (∀ tag : Z,
     existsb (Z.eqb tag) seen = false ->
     shape_of_tag tag all = shape_of_tag tag cur) ->
  shapes_say_at_least seen cur all = true
with fields_say_at_least_refl (sks : field_subkinds) {struct sks} :
  fields_say_at_least sks sks = true.
Proof.
  - destruct a as [ | | n | consts shapes]; cbn [subkind_says_at_least];
      try reflexivity.
    + apply boxable_number_eqb_true. reflexivity.
    + apply andb_true_iff. split.
      * apply forallb_forall. intros i Hi. apply existsb_eqb_In. exact Hi.
      * apply shapes_say_at_least_refl. intros tag _. reflexivity.
  - destruct cur as [ | tag0 rest | tag0 sks rest];
      cbn [shapes_say_at_least]; intros Hall; [ reflexivity | | ];
      (apply andb_true_iff; split;
       [ destruct (existsb (Z.eqb tag0) seen) eqn:Hseen; [ reflexivity | ];
         cbn [orb]; rewrite (Hall tag0 Hseen); cbn [shape_of_tag];
         rewrite Z.eqb_refl
       | apply shapes_say_at_least_refl; intros tag Htag;
         cbn [existsb] in Htag; apply orb_false_iff in Htag;
         destruct Htag as [Hne Hseen]; rewrite (Hall tag Hseen);
         cbn [shape_of_tag]; rewrite Hne; reflexivity ]).
    + reflexivity.
    + apply fields_say_at_least_refl.
  - destruct sks as [ | nullable sk rest]; cbn [fields_say_at_least];
      [ reflexivity | ].
    rewrite subkind_says_at_least_refl, fields_say_at_least_refl.
    destruct nullable; reflexivity.
Qed.

Lemma param_says_at_least_refl (a : param_subkind) :
  param_says_at_least a a = true.
Proof.
  destruct a as [n sk]. unfold param_says_at_least. cbn [fst snd].
  rewrite subkind_says_at_least_refl. destruct n; reflexivity.
Qed.

Theorem params_say_at_least_refl : params_say_at_least_refl_statement.
Proof.
  intros a. induction a as [ | pa a IH]; [ reflexivity | ].
  cbn [params_say_at_least]. rewrite param_says_at_least_refl, IH.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Transitive                                                               *)
(* ======================================================================== *)

(* What the order says of the first entry for a tag that is not among
   those gone past. *)
Lemma shapes_say_at_least_at (seen : list Z)
    (shapes shapes' : constructor_shapes) (tag : Z) :
  shapes_say_at_least seen shapes shapes' = true ->
  existsb (Z.eqb tag) seen = false ->
  match shape_of_tag tag shapes with
  | None => True
  | Some None => shape_of_tag tag shapes' = Some None
  | Some (Some sks) =>
    match shape_of_tag tag shapes' with
    | Some None => True
    | Some (Some sks') => fields_say_at_least sks sks' = true
    | None => False
    end
  end.
Proof.
  revert seen.
  induction shapes as [ | tag0 rest IH | tag0 sks rest IH]; intros seen;
    cbn [shapes_say_at_least shape_of_tag]; intros Hle Hseen;
    [ exact I | | ];
    apply andb_true_iff in Hle; destruct Hle as [Hhead Hrest];
    (destruct (Z.eqb tag tag0) eqn:Htag;
     [ apply Z.eqb_eq in Htag; subst tag0; rewrite Hseen in Hhead;
       cbn [orb] in Hhead
     | apply (IH (tag0 :: seen) Hrest); cbn [existsb];
       rewrite Htag, Hseen; reflexivity ]).
  - destruct (shape_of_tag tag shapes') as [[sks' | ] | ];
      try discriminate Hhead.
    reflexivity.
  - destruct (shape_of_tag tag shapes') as [[sks' | ] | ];
      [ exact Hhead | exact I | discriminate Hhead ].
Qed.

Fixpoint subkind_says_at_least_trans (a b c : subkind) {struct a} :
  subkind_says_at_least a b = true ->
  subkind_says_at_least b c = true ->
  subkind_says_at_least a c = true
with shapes_say_at_least_trans (seen : list Z)
    (shapes shapes' shapes'' : constructor_shapes) {struct shapes} :
  shapes_say_at_least seen shapes shapes' = true ->
  shapes_say_at_least [] shapes' shapes'' = true ->
  shapes_say_at_least seen shapes shapes'' = true
with fields_say_at_least_trans (sks sks' sks'' : field_subkinds)
    {struct sks} :
  fields_say_at_least sks sks' = true ->
  fields_say_at_least sks' sks'' = true ->
  fields_say_at_least sks sks'' = true.
Proof.
  - destruct c as [ | | n'' | consts'' shapes''].
    + intros _ _. destruct a; reflexivity.
    + destruct b as [ | | n' | consts' shapes'];
        cbn [subkind_says_at_least]; intros Hab Hbc; try discriminate Hbc.
      * exact Hab.
      * destruct shapes'; try discriminate Hbc.
        destruct a as [ | | n | consts shapes];
          cbn [subkind_says_at_least] in Hab |- *; try discriminate Hab.
        apply andb_true_iff in Hab. destruct Hab as [Hconsts Hshapes].
        destruct shapes as [ | tag0 rest | tag0 sks rest];
          [ | cbn [shapes_say_at_least existsb orb shape_of_tag andb]
                in Hshapes;
              discriminate Hshapes.. ].
        reflexivity.
    + destruct b as [ | | n' | consts' shapes'];
        cbn [subkind_says_at_least]; intros Hab Hbc; try discriminate Hbc.
      apply boxable_number_eqb_true in Hbc. subst n''. exact Hab.
    + destruct b as [ | | n' | consts' shapes'];
        cbn [subkind_says_at_least]; intros Hab Hbc; try discriminate Hbc.
      destruct a as [ | | n | consts shapes];
        cbn [subkind_says_at_least] in Hab |- *; try discriminate Hab.
      apply andb_true_iff in Hab. destruct Hab as [Hconsts Hshapes].
      apply andb_true_iff in Hbc. destruct Hbc as [Hconsts' Hshapes'].
      apply andb_true_iff. split.
      * apply forallb_forall. intros i Hi.
        apply (consts_among consts' consts'' i Hconsts').
        apply (consts_among consts consts' i Hconsts).
        apply existsb_eqb_In. exact Hi.
      * exact
          (shapes_say_at_least_trans [] shapes shapes' shapes'' Hshapes
             Hshapes').
  - destruct shapes as [ | tag0 rest | tag0 sks rest];
      cbn [shapes_say_at_least]; intros Hab Hbc; [ reflexivity | | ];
      apply andb_true_iff in Hab; destruct Hab as [Hhead Hrest];
      (apply andb_true_iff; split;
       [ | exact
             (shapes_say_at_least_trans (tag0 :: seen) rest shapes' shapes''
                Hrest Hbc) ]);
      (destruct (existsb (Z.eqb tag0) seen); [ reflexivity | ]);
      cbn [orb] in Hhead |- *;
      pose proof (shapes_say_at_least_at [] shapes' shapes'' tag0 Hbc eq_refl)
        as Hat;
      revert Hhead Hat;
      destruct (shape_of_tag tag0 shapes') as [[sks' | ] | ];
      intros Hhead Hat; try discriminate Hhead.
    + rewrite Hat. reflexivity.
    + revert Hat.
      destruct (shape_of_tag tag0 shapes'') as [[sks'' | ] | ]; intros Hat;
        [ exact (fields_say_at_least_trans sks sks' sks'' Hhead Hat)
        | reflexivity
        | contradiction ].
    + rewrite Hat. reflexivity.
  - destruct sks as [ | n sk rest]; destruct sks' as [ | n' sk' rest'];
      cbn [fields_say_at_least]; intros Hab Hbc; try discriminate Hab.
    + exact Hbc.
    + destruct sks'' as [ | n'' sk'' rest''];
        cbn [fields_say_at_least] in Hbc |- *; try discriminate Hbc.
      apply andb_true_iff in Hab. destruct Hab as [Hab Hrest].
      apply andb_true_iff in Hab. destruct Hab as [Hn Hsk].
      apply andb_true_iff in Hbc. destruct Hbc as [Hbc Hrest'].
      apply andb_true_iff in Hbc. destruct Hbc as [Hn' Hsk'].
      rewrite (subkind_says_at_least_trans sk sk' sk'' Hsk Hsk'),
        (fields_say_at_least_trans rest rest' rest'' Hrest Hrest').
      destruct n, n', n''; cbn in Hn, Hn' |- *; congruence.
Qed.

Lemma param_says_at_least_trans (a b c : param_subkind) :
  param_says_at_least a b = true ->
  param_says_at_least b c = true ->
  param_says_at_least a c = true.
Proof.
  destruct a as [na ska], b as [nb skb], c as [nc skc].
  unfold param_says_at_least. cbn [fst snd]. intros Hab Hbc.
  apply andb_true_iff in Hab. destruct Hab as [Hn Hsk].
  apply andb_true_iff in Hbc. destruct Hbc as [Hn' Hsk'].
  rewrite (subkind_says_at_least_trans ska skb skc Hsk Hsk').
  destruct na, nb, nc; cbn in Hn, Hn' |- *; congruence.
Qed.

(* Of what says nothing, only what says nothing says no more; and
   anything says at least as much as it. *)
Lemma says_nothing_only (b c : param_subkind) :
  says_nothing (fst b) (snd b) = true ->
  param_says_at_least b c = true ->
  says_nothing (fst c) (snd c) = true.
Proof.
  destruct b as [nb skb], c as [nc skc]. unfold param_says_at_least.
  cbn [fst snd]. intros Hb Hle. apply andb_true_iff in Hle.
  destruct Hle as [Hn Hsk].
  exact (says_nothing_weaken nb skb nc skc Hn Hsk Hb).
Qed.

Lemma says_nothing_above (a c : param_subkind) :
  says_nothing (fst c) (snd c) = true -> param_says_at_least a c = true.
Proof.
  destruct a as [na ska], c as [nc skc]. cbn [fst snd].
  unfold says_nothing, param_says_at_least. cbn [fst snd]. intros H.
  apply andb_true_iff in H. destruct H as [Hn Hany].
  destruct skc; try discriminate Hany. subst nc.
  destruct na, ska; reflexivity.
Qed.

Theorem params_say_at_least_trans : params_say_at_least_trans_statement.
Proof.
  intros a b c. revert a b.
  induction c as [ | pc c IH]; intros a b Hab Hbc; [ reflexivity | ].
  destruct b as [ | pb b]; destruct a as [ | pa a];
    cbn [params_say_at_least] in Hab, Hbc |- *.
  - exact Hbc.
  - apply andb_true_iff in Hbc. destruct Hbc as [Hnothing Hrest].
    rewrite (says_nothing_above pa pc Hnothing), (IH a [] eq_refl Hrest).
    reflexivity.
  - apply andb_true_iff in Hab. destruct Hab as [Hnothing Hrest].
    apply andb_true_iff in Hbc. destruct Hbc as [Hhead Hrest'].
    rewrite (says_nothing_only pb pc Hnothing Hhead), (IH [] b Hrest Hrest').
    reflexivity.
  - apply andb_true_iff in Hab. destruct Hab as [Hhead Hrest].
    apply andb_true_iff in Hbc. destruct Hbc as [Hhead' Hrest'].
    rewrite (param_says_at_least_trans pa pb pc Hhead Hhead'),
      (IH a b Hrest Hrest').
    reflexivity.
Qed.
