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

(* Proofs of the statements of FlambdaLocationsSpec.v. Nothing here
   needs review.

   Each action is shown to have four laws ([action_laws]): the
   identity, composition, what it does to the locations of a thing,
   and that two permutations that agree on the locations of a thing
   move it alike. The laws of FlambdaNameAbstractionSpec.nominal
   follow from the four ([nominal_of_laws]). They are kept by lists,
   by pairs and by bindings whose keys are not locations; the values
   and the continuations, which hold themselves, are by induction. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaEquivarianceSpec FlambdaHeapSpec FlambdaHeapProofs
  FlambdaHeapMachineProofs FlambdaLocationsSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* The laws of an action                                                    *)
(* ======================================================================== *)

Record action_laws {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) : Prop := Action_laws {
  act_identity : ∀ t : T, act identity_permutation t = t;
  act_compose :
    ∀ (second first : permutation) (t : T),
      act (compose_permutation second first) t = act second (act first t);
  act_names :
    ∀ (p : permutation) (t : T), names (act p t) = map (permute p) (names t);
  act_ext :
    ∀ (p p' : permutation) (t : T),
      (∀ name : positive,
         name ∈ names t -> permute p name = permute p' name) ->
      act p t = act p' t
}.

Ltac use_laws H :=
  destruct H as [Hidentity Hcompose Hnames Hext];
  cbv beta in Hidentity, Hcompose, Hnames, Hext.

Lemma act_same {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) (p p' : permutation) (t : T) :
  action_laws act names -> same_permutation p p' -> act p t = act p' t.
Proof.
  intros Hlaws Hsame. apply (act_ext _ _ Hlaws). intros name _. apply Hsame.
Qed.

Lemma act_fixes {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) (p : permutation) (t : T) :
  action_laws act names -> fixes p (names t) -> act p t = t.
Proof.
  intros Hlaws Hfixes. rewrite <- (act_identity _ _ Hlaws t) at 2.
  apply (act_ext _ _ Hlaws). intros name Hname. apply Hfixes, among_In, Hname.
Qed.

Lemma act_inverse {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) (p : permutation) (t : T) :
  action_laws act names ->
  act (inverse_permutation p) (act p t) = t
  ∧ act p (act (inverse_permutation p) t) = t.
Proof.
  intros Hlaws. split.
  - rewrite <- (act_compose _ _ Hlaws).
    rewrite <- (act_identity _ _ Hlaws t) at 2.
    apply (act_same _ _ _ _ _ Hlaws). intros name.
    unfold compose_permutation, inverse_permutation, identity_permutation.
    rewrite permute_app. apply permute_rev_l.
  - rewrite <- (act_compose _ _ Hlaws).
    rewrite <- (act_identity _ _ Hlaws t) at 2.
    apply (act_same _ _ _ _ _ Hlaws). intros name.
    unfold compose_permutation, inverse_permutation, identity_permutation.
    rewrite permute_app. apply permute_rev_r.
Qed.

Lemma nominal_of_laws {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) :
  action_laws act names -> nominal (Contains_names act names eq).
Proof.
  intros Hlaws. constructor; simpl.
  - reflexivity.
  - intros t t' ->. reflexivity.
  - intros t t' t'' -> ->. reflexivity.
  - apply (act_identity _ _ Hlaws).
  - apply (act_compose _ _ Hlaws).
  - intros p p' t Hsame. apply (act_same _ _ _ _ _ Hlaws Hsame).
  - intros p t t' ->. reflexivity.
  - intros p t name. rewrite (act_names _ _ Hlaws). apply among_permuted.
  - intros t t' -> name. reflexivity.
  - intros p t Hfixes. apply (act_fixes _ _ _ _ Hlaws Hfixes).
Qed.

(* A location. *)
Lemma laws_of_name : action_laws permute (fun name => [name]).
Proof.
  constructor.
  - reflexivity.
  - intros second first name. apply permute_app.
  - reflexivity.
  - intros p p' name Hagree. apply Hagree. left. reflexivity.
Qed.

(* Pairs. *)
Lemma laws_of_pair {A B : Type} (actA : permutation -> A -> A)
    (namesA : A -> list positive) (actB : permutation -> B -> B)
    (namesB : B -> list positive) :
  action_laws actA namesA ->
  action_laws actB namesB ->
  action_laws (fun p ab => (actA p (fst ab), actB p (snd ab)))
    (fun ab => namesA (fst ab) ++ namesB (snd ab)).
Proof.
  intros HA HB. constructor.
  - intros [a b]. simpl.
    rewrite (act_identity _ _ HA), (act_identity _ _ HB). reflexivity.
  - intros second first [a b]. simpl.
    rewrite (act_compose _ _ HA), (act_compose _ _ HB). reflexivity.
  - intros p [a b]. simpl.
    rewrite map_app, (act_names _ _ HA), (act_names _ _ HB). reflexivity.
  - intros p p' [a b] Hagree. simpl in *. f_equal.
    + apply (act_ext _ _ HA). intros name Hname. apply Hagree, in_or_app.
      left. exact Hname.
    + apply (act_ext _ _ HB). intros name Hname. apply Hagree, in_or_app.
      right. exact Hname.
Qed.

(* A thing under a key that is not a location. *)
Lemma laws_of_keyed {K T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) :
  action_laws act names ->
  action_laws (fun p (b : K * T) => (fst b, act p (snd b)))
    (fun b : K * T => names (snd b)).
Proof.
  intros Hlaws. constructor.
  - intros [k t]. simpl. rewrite (act_identity _ _ Hlaws). reflexivity.
  - intros second first [k t]. simpl. rewrite (act_compose _ _ Hlaws).
    reflexivity.
  - intros p [k t]. simpl. apply (act_names _ _ Hlaws).
  - intros p p' [k t] Hagree. simpl in *. f_equal.
    apply (act_ext _ _ Hlaws), Hagree.
Qed.

(* Lists. *)
Lemma laws_of_list {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) :
  action_laws act names ->
  action_laws (fun p => map (act p)) (flat_map names).
Proof.
  intros Hlaws. constructor.
  - intros l. induction l as [ | t l IH]; simpl; [ reflexivity | ].
    rewrite (act_identity _ _ Hlaws), IH. reflexivity.
  - intros second first l. rewrite map_map. apply map_ext. intros t.
    apply (act_compose _ _ Hlaws).
  - intros p l. induction l as [ | t l IH]; simpl; [ reflexivity | ].
    rewrite map_app, (act_names _ _ Hlaws), IH. reflexivity.
  - intros p p' l Hagree. induction l as [ | t l IH]; simpl in *;
      [ reflexivity | ].
    f_equal.
    + apply (act_ext _ _ Hlaws). intros name Hname. apply Hagree, in_or_app.
      left. exact Hname.
    + apply IH. intros name Hname. apply Hagree, in_or_app. right.
      exact Hname.
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* Maps over the values that a value holds, given what holds of each. *)
Lemma map_Forall_ext {A B : Type} (f g : A -> B) (l : list A) :
  Forall (fun a => f a = g a) l -> map f l = map g l.
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; [ reflexivity | ].
  rewrite Ha, IH. reflexivity.
Qed.

Lemma map_slots_Forall_ext {K A : Type} (f g : A -> A) (l : list (K * A)) :
  Forall (fun slot => f (snd slot) = g (snd slot)) l ->
  map (fun slot => (fst slot, f (snd slot))) l
  = map (fun slot => (fst slot, g (snd slot))) l.
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; [ reflexivity | ].
  rewrite Ha, IH. reflexivity.
Qed.

Lemma map_slots_id {K A : Type} (l : list (K * A)) :
  map (fun slot => (fst slot, snd slot)) l = l.
Proof.
  induction l as [ | [k a] l IH]; simpl; [ reflexivity | ]. rewrite IH.
  reflexivity.
Qed.

Lemma map_slots_map_slots {K A : Type} (f g : A -> A) (l : list (K * A)) :
  map (fun slot => (fst slot, g (snd slot)))
    (map (fun slot => (fst slot, f (snd slot))) l)
  = map (fun slot => (fst slot, g (f (snd slot)))) l.
Proof. rewrite map_map. reflexivity. Qed.

Lemma flat_map_map_Forall {A : Type} (act : A -> A)
    (names : A -> list positive) (f : positive -> positive) (l : list A) :
  Forall (fun a => names (act a) = map f (names a)) l ->
  flat_map names (map act l) = map f (flat_map names l).
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; [ reflexivity | ].
  rewrite map_app, Ha, IH. reflexivity.
Qed.

Lemma flat_map_slots_Forall {K A : Type} (act : A -> A)
    (names : A -> list positive) (f : positive -> positive)
    (l : list (K * A)) :
  Forall (fun slot => names (act (snd slot)) = map f (names (snd slot))) l ->
  flat_map (fun slot => names (snd slot))
    (map (fun slot => (fst slot, act (snd slot))) l)
  = map f (flat_map (fun slot => names (snd slot)) l).
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; [ reflexivity | ].
  rewrite map_app, Ha, IH. reflexivity.
Qed.

Lemma map_agree {A B : Type} (f g : A -> B) (names : A -> list positive)
    (P : positive -> Prop) (l : list A) :
  Forall (fun a => (∀ name : positive, name ∈ names a -> P name) -> f a = g a)
    l ->
  (∀ name : positive, name ∈ flat_map names l -> P name) ->
  map f l = map g l.
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; intros Hagree;
    [ reflexivity | ].
  rewrite Ha, IH; [ reflexivity | | ]; intros name Hname;
    apply Hagree, in_or_app; [ right | left ]; exact Hname.
Qed.

Lemma map_slots_agree {K A : Type} (f g : A -> A)
    (names : A -> list positive) (P : positive -> Prop) (l : list (K * A)) :
  Forall
    (fun slot =>
       (∀ name : positive, name ∈ names (snd slot) -> P name) ->
       f (snd slot) = g (snd slot))
    l ->
  (∀ name : positive,
     name ∈ flat_map (fun slot => names (snd slot)) l -> P name) ->
  map (fun slot => (fst slot, f (snd slot))) l
  = map (fun slot => (fst slot, g (snd slot))) l.
Proof.
  intros Hall. induction Hall as [ | a l Ha _ IH]; simpl; intros Hagree;
    [ reflexivity | ].
  rewrite Ha, IH; [ reflexivity | | ]; intros name Hname;
    apply Hagree, in_or_app; [ right | left ]; exact Hname.
Qed.

Lemma permute_value_identity (v : value) :
  permute_value identity_permutation v = v.
Proof.
  induction v as [c | code slots IH | slot decls slots IH | s | tag fields IH
                 | elements IH | c | l
                 | callee args args_arity IHcallee IHargs]
    using value_induction; simpl; try reflexivity.
  - rewrite (map_slots_Forall_ext _ (fun v => v) slots IH), map_slots_id.
    reflexivity.
  - rewrite (map_slots_Forall_ext _ (fun v => v) slots IH), map_slots_id.
    reflexivity.
  - rewrite (map_Forall_ext _ (fun v => v) fields IH), map_id. reflexivity.
  - rewrite (map_Forall_ext _ (fun v => v) elements IH), map_id.
    reflexivity.
  - rewrite IHcallee, (map_Forall_ext _ (fun v => v) args IHargs), map_id.
    reflexivity.
Qed.

Lemma permute_value_compose (second first : permutation) (v : value) :
  permute_value (compose_permutation second first) v
  = permute_value second (permute_value first v).
Proof.
  induction v as [c | code slots IH | slot decls slots IH | s | tag fields IH
                 | elements IH | c | l
                 | callee args args_arity IHcallee IHargs]
    using value_induction; simpl; try reflexivity.
  - rewrite map_slots_map_slots.
    rewrite (map_slots_Forall_ext _
               (fun v => permute_value second (permute_value first v)) slots
               IH).
    reflexivity.
  - rewrite map_slots_map_slots.
    rewrite (map_slots_Forall_ext _
               (fun v => permute_value second (permute_value first v)) slots
               IH).
    reflexivity.
  - rewrite map_map.
    rewrite (map_Forall_ext _
               (fun v => permute_value second (permute_value first v)) fields
               IH).
    reflexivity.
  - rewrite map_map.
    rewrite (map_Forall_ext _
               (fun v => permute_value second (permute_value first v))
               elements IH).
    reflexivity.
  - unfold compose_permutation. rewrite permute_app. reflexivity.
  - rewrite IHcallee, map_map.
    rewrite (map_Forall_ext _
               (fun v => permute_value second (permute_value first v)) args
               IHargs).
    reflexivity.
Qed.

Lemma locations_of_permute_value (p : permutation) (v : value) :
  locations_of_value (permute_value p v)
  = map (permute p) (locations_of_value v).
Proof.
  induction v as [c | code slots IH | slot decls slots IH | s | tag fields IH
                 | elements IH | c | l
                 | callee args args_arity IHcallee IHargs]
    using value_induction; simpl; try reflexivity.
  - apply (flat_map_slots_Forall (permute_value p) locations_of_value), IH.
  - apply (flat_map_slots_Forall (permute_value p) locations_of_value), IH.
  - apply (flat_map_map_Forall (permute_value p) locations_of_value), IH.
  - apply (flat_map_map_Forall (permute_value p) locations_of_value), IH.
  - rewrite map_app, IHcallee. f_equal.
    exact (flat_map_map_Forall (permute_value p) locations_of_value
             (permute p) args IHargs).
Qed.

Lemma permute_value_ext (p p' : permutation) (v : value) :
  (∀ l : location, l ∈ locations_of_value v -> permute p l = permute p' l) ->
  permute_value p v = permute_value p' v.
Proof.
  induction v as [c | code slots IH | slot decls slots IH | s | tag fields IH
                 | elements IH | c | l
                 | callee args args_arity IHcallee IHargs]
    using value_induction; simpl; intros Hagree; try reflexivity.
  - rewrite (map_slots_agree _ _ locations_of_value _ slots IH Hagree).
    reflexivity.
  - rewrite (map_slots_agree _ _ locations_of_value _ slots IH Hagree).
    reflexivity.
  - rewrite (map_agree _ _ locations_of_value _ fields IH Hagree).
    reflexivity.
  - rewrite (map_agree _ _ locations_of_value _ elements IH Hagree).
    reflexivity.
  - rewrite Hagree; [ reflexivity | left; reflexivity ].
  - rewrite IHcallee, (map_agree _ _ locations_of_value _ args IHargs);
      [ reflexivity | | ]; intros l Hl; apply Hagree, in_or_app;
      [ right | left ]; exact Hl.
Qed.

Lemma value_laws : action_laws permute_value locations_of_value.
Proof.
  constructor.
  - exact permute_value_identity.
  - exact permute_value_compose.
  - exact locations_of_permute_value.
  - exact permute_value_ext.
Qed.

Lemma values_laws :
  action_laws (fun p => map (permute_value p)) (flat_map locations_of_value).
Proof. exact (laws_of_list _ _ value_laws). Qed.

Lemma slots_laws {K : Type} :
  action_laws
    (fun p =>
       map (fun slot : K * value => (fst slot, permute_value p (snd slot))))
    (flat_map (fun slot : K * value => locations_of_value (snd slot))).
Proof. exact (laws_of_list _ _ (laws_of_keyed _ _ value_laws)). Qed.

(* ======================================================================== *)
(* Cells, heaps, environments, the static environment                       *)
(* ======================================================================== *)

Lemma flat_map_map_snd {K A B : Type} (f : A -> list B) (l : list (K * A)) :
  flat_map f (map snd l) = flat_map (fun slot => f (snd slot)) l.
Proof.
  induction l as [ | [k a] l IH]; simpl; [ reflexivity | ]. rewrite IH.
  reflexivity.
Qed.

Lemma cell_laws : action_laws permute_cell locations_of_cell.
Proof.
  pose proof values_laws as Hvalues. use_laws Hvalues. constructor.
  - intros [mut tag fields | kind elements]; simpl; rewrite Hidentity;
      reflexivity.
  - intros second first [mut tag fields | kind elements]; simpl;
      rewrite Hcompose; reflexivity.
  - intros p [mut tag fields | kind elements]; apply Hnames.
  - intros p p' [mut tag fields | kind elements] Hagree; simpl; f_equal;
      apply Hext, Hagree.
Qed.

Lemma heap_laws : action_laws permute_heap locations_of_heap.
Proof.
  exact (laws_of_list _ _ (laws_of_pair _ _ _ _ laws_of_name cell_laws)).
Qed.

Lemma env_laws : action_laws permute_env locations_of_env.
Proof. exact (laws_of_list _ _ (laws_of_keyed _ _ value_laws)). Qed.

Lemma static_value_laws :
  action_laws permute_static_value locations_of_static_value.
Proof.
  pose proof values_laws as Hvalues. use_laws Hvalues.
  constructor.
  - intros [code slots | slot closure_symbols decls slots | tag fields
           | contents | bytes | elements | class]; simpl.
    + rewrite (act_identity _ _ slots_laws). reflexivity.
    + rewrite (act_identity _ _ slots_laws). reflexivity.
    + rewrite Hidentity. reflexivity.
    + rewrite permute_value_identity. reflexivity.
    + reflexivity.
    + rewrite Hidentity. reflexivity.
    + reflexivity.
  - intros second first
      [code slots | slot closure_symbols decls slots | tag fields | contents
      | bytes | elements | class];
      simpl.
    + rewrite (act_compose _ _ slots_laws). reflexivity.
    + rewrite (act_compose _ _ slots_laws). reflexivity.
    + rewrite Hcompose. reflexivity.
    + rewrite permute_value_compose. reflexivity.
    + reflexivity.
    + rewrite Hcompose. reflexivity.
    + reflexivity.
  - intros p
      [code slots | slot closure_symbols decls slots | tag fields | contents
      | bytes | elements | class];
      unfold locations_of_static_value; simpl.
    + rewrite !flat_map_map_snd.
      apply (act_names _ _ (@slots_laws value_slot)).
    + rewrite !flat_map_map_snd.
      apply (act_names _ _ (@slots_laws value_slot)).
    + apply Hnames.
    + rewrite !app_nil_r. apply locations_of_permute_value.
    + reflexivity.
    + apply Hnames.
    + reflexivity.
  - intros p p'
      [code slots | slot closure_symbols decls slots | tag fields | contents
      | bytes | elements | class]
      Hagree;
      unfold locations_of_static_value in Hagree; simpl in *; f_equal.
    + apply (act_ext _ _ (@slots_laws value_slot)). intros name Hname.
      apply Hagree. rewrite flat_map_map_snd. exact Hname.
    + apply (act_ext _ _ (@slots_laws value_slot)). intros name Hname.
      apply Hagree. rewrite flat_map_map_snd. exact Hname.
    + apply Hext, Hagree.
    + apply permute_value_ext. intros l Hl. apply Hagree.
      rewrite app_nil_r. exact Hl.
    + apply Hext, Hagree.
Qed.

Lemma static_values_laws :
  action_laws
    (fun p => map_values (permute_static_value p))
    (flat_map
       (fun binding : symbol * statically_allocated_value =>
          locations_of_static_value (snd binding))).
Proof. exact (laws_of_list _ _ (laws_of_keyed _ _ static_value_laws)). Qed.

Lemma static_env_laws :
  action_laws permute_static_env locations_of_static_env.
Proof.
  pose proof static_values_laws as Hvalues. use_laws Hvalues. constructor.
  - intros [codes values]. unfold permute_static_env. simpl.
    rewrite Hidentity. reflexivity.
  - intros second first [codes values]. unfold permute_static_env. simpl.
    rewrite Hcompose. reflexivity.
  - intros p [codes values]. apply Hnames.
  - intros p p' [codes values] Hagree. unfold permute_static_env. simpl.
    f_equal. apply Hext, Hagree.
Qed.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

Lemma permute_cont_value_identity (c : cont_value) :
  permute_cont_value identity_permutation c = c.
Proof.
  induction c as [h vars conts recursive IH | k | c extra IH
                 | remaining remaining_arity return_arity return_to exn_to
                     IHreturn IHexn]
    using cont_value_induction; simpl.
  - rewrite (act_identity _ _ env_laws).
    rewrite (map_slots_Forall_ext _ (fun c => c) conts IH), map_slots_id.
    reflexivity.
  - reflexivity.
  - rewrite IH, (act_identity _ _ values_laws). reflexivity.
  - rewrite IHreturn, IHexn, (act_identity _ _ values_laws). reflexivity.
Qed.

Lemma permute_cont_value_compose (second first : permutation)
    (c : cont_value) :
  permute_cont_value (compose_permutation second first) c
  = permute_cont_value second (permute_cont_value first c).
Proof.
  induction c as [h vars conts recursive IH | k | c extra IH
                 | remaining remaining_arity return_arity return_to exn_to
                     IHreturn IHexn]
    using cont_value_induction; simpl.
  - rewrite (act_compose _ _ env_laws), map_slots_map_slots.
    rewrite (map_slots_Forall_ext _
               (fun c =>
                  permute_cont_value second (permute_cont_value first c))
               conts IH).
    reflexivity.
  - reflexivity.
  - rewrite IH, (act_compose _ _ values_laws). reflexivity.
  - rewrite IHreturn, IHexn, (act_compose _ _ values_laws). reflexivity.
Qed.

Lemma locations_of_permute_cont_value (p : permutation) (c : cont_value) :
  locations_of_cont_value (permute_cont_value p c)
  = map (permute p) (locations_of_cont_value c).
Proof.
  induction c as [h vars conts recursive IH | k | c extra IH
                 | remaining remaining_arity return_arity return_to exn_to
                     IHreturn IHexn]
    using cont_value_induction; simpl.
  - etransitivity; [ | symmetry; apply map_app ]. f_equal.
    + apply (act_names _ _ env_laws).
    + exact (flat_map_slots_Forall (permute_cont_value p)
               locations_of_cont_value (permute p) conts IH).
  - reflexivity.
  - etransitivity; [ | symmetry; apply map_app ]. f_equal.
    + exact IH.
    + apply (act_names _ _ values_laws).
  - etransitivity; [ | symmetry; apply map_app ]. f_equal.
    + apply (act_names _ _ values_laws).
    + etransitivity; [ | symmetry; apply map_app ]. f_equal; assumption.
Qed.

Lemma permute_cont_value_ext (p p' : permutation) (c : cont_value) :
  (∀ l : location,
     l ∈ locations_of_cont_value c -> permute p l = permute p' l) ->
  permute_cont_value p c = permute_cont_value p' c.
Proof.
  induction c as [h vars conts recursive IH | k | c extra IH
                 | remaining remaining_arity return_arity return_to exn_to
                     IHreturn IHexn]
    using cont_value_induction; simpl; intros Hagree.
  - rewrite (act_ext _ _ env_laws p p' vars).
    + rewrite (map_slots_agree _ _ locations_of_cont_value _ conts IH);
        [ reflexivity | ].
      intros l Hl. apply Hagree, in_or_app. right. exact Hl.
    + intros l Hl. apply Hagree, in_or_app. left. exact Hl.
  - reflexivity.
  - rewrite IH, (act_ext _ _ values_laws p p' extra); [ reflexivity | | ];
      intros l Hl; apply Hagree, in_or_app; [ right | left ]; exact Hl.
  - rewrite IHreturn, IHexn, (act_ext _ _ values_laws p p' remaining);
      [ reflexivity | | | ]; intros l Hl; apply Hagree, in_or_app.
    + left. exact Hl.
    + right. apply in_or_app. right. exact Hl.
    + right. apply in_or_app. left. exact Hl.
Qed.

Lemma cont_value_laws :
  action_laws permute_cont_value locations_of_cont_value.
Proof.
  constructor.
  - exact permute_cont_value_identity.
  - exact permute_cont_value_compose.
  - exact locations_of_permute_cont_value.
  - exact permute_cont_value_ext.
Qed.

Lemma cont_env_laws : action_laws permute_cont_env locations_of_cont_env.
Proof. exact (laws_of_list _ _ (laws_of_keyed _ _ cont_value_laws)). Qed.

(* The read record: the locations of its entries are moved. *)
Lemma reads_laws : action_laws permute_reads locations_of_reads.
Proof.
  constructor.
  - intros rd. unfold permute_reads.
    induction rd as [ | [l extent] rd IH]; [ reflexivity | ].
    cbn [map fst snd]. rewrite (act_identity _ _ laws_of_name), IH.
    reflexivity.
  - intros second first rd. unfold permute_reads.
    induction rd as [ | [l extent] rd IH]; [ reflexivity | ].
    cbn [map fst snd]. rewrite (act_compose _ _ laws_of_name), IH.
    reflexivity.
  - intros p rd. unfold permute_reads, locations_of_reads.
    rewrite !map_map. reflexivity.
  - intros p p' rd Hagree. unfold permute_reads. apply map_ext_in.
    intros [l extent] Hin. simpl. rewrite (Hagree l); [ reflexivity | ].
    unfold locations_of_reads. apply in_map_iff. exists (l, extent).
    split; [ reflexivity | exact Hin ].
Qed.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

Lemma state_laws : action_laws permute_state locations_of_state.
Proof.
  constructor.
  - intros [e vars conts statics hp rd]. unfold permute_state. simpl.
    rewrite (act_identity _ _ env_laws), (act_identity _ _ cont_env_laws),
      (act_identity _ _ static_env_laws), (act_identity _ _ heap_laws),
      (act_identity _ _ reads_laws).
    reflexivity.
  - intros second first [e vars conts statics hp rd]. unfold permute_state.
    simpl.
    rewrite (act_compose _ _ env_laws), (act_compose _ _ cont_env_laws),
      (act_compose _ _ static_env_laws), (act_compose _ _ heap_laws),
      (act_compose _ _ reads_laws).
    reflexivity.
  - intros p [e vars conts statics hp rd]. unfold locations_of_state. simpl.
    rewrite !map_app, (act_names _ _ env_laws),
      (act_names _ _ cont_env_laws), (act_names _ _ static_env_laws),
      (act_names _ _ heap_laws), (act_names _ _ reads_laws).
    reflexivity.
  - intros p p' [e vars conts statics hp rd] Hagree.
    unfold locations_of_state in Hagree. unfold permute_state. simpl in *.
    rewrite (act_ext _ _ env_laws p p' vars),
      (act_ext _ _ cont_env_laws p p' conts),
      (act_ext _ _ static_env_laws p p' statics),
      (act_ext _ _ heap_laws p p' hp), (act_ext _ _ reads_laws p p' rd);
      [ reflexivity | | | | | ]; intros l Hl; apply Hagree, in_or_app.
    + right. apply in_or_app. right. apply in_or_app. right.
      apply in_or_app. right. exact Hl.
    + right. apply in_or_app. right. apply in_or_app. right.
      apply in_or_app. left. exact Hl.
    + right. apply in_or_app. right. apply in_or_app. left. exact Hl.
    + right. apply in_or_app. left. exact Hl.
    + left. exact Hl.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem value_locations_nominal : value_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ value_laws). Qed.

Theorem cell_locations_nominal : cell_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ cell_laws). Qed.

Theorem heap_locations_nominal : heap_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ heap_laws). Qed.

Theorem env_locations_nominal : env_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ env_laws). Qed.

Theorem static_env_locations_nominal :
  static_env_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ static_env_laws). Qed.

Theorem cont_value_locations_nominal :
  cont_value_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ cont_value_laws). Qed.

Theorem state_locations_nominal : state_locations_nominal_statement.
Proof. exact (nominal_of_laws _ _ state_laws). Qed.

Theorem permute_state_inverse : permute_state_inverse_statement.
Proof. intros p s. exact (act_inverse _ _ p s state_laws). Qed.

(* The two actions. *)

Lemma rename_permute_env (r : renaming) (p : permutation) (vars : env) :
  rename_env r (permute_env p vars) = permute_env p (rename_env r vars).
Proof.
  unfold rename_env, permute_env, map_values. rewrite !map_map.
  apply map_ext. intros [x v]. reflexivity.
Qed.

Lemma rename_permute_cont_value (r : renaming) (p : permutation)
    (c : cont_value) :
  rename_cont_value r (permute_cont_value p c)
  = permute_cont_value p (rename_cont_value r c).
Proof.
  induction c as [h vars conts recursive IH | k | c extra IH
                 | remaining remaining_arity return_arity return_to exn_to
                     IHreturn IHexn]
    using cont_value_induction; simpl.
  - rewrite rename_permute_env. f_equal. rewrite !map_map.
    induction IH as [ | [k c] conts Hc _ IHconts]; simpl; [ reflexivity | ].
    simpl in Hc. rewrite Hc, IHconts. reflexivity.
  - reflexivity.
  - rewrite IH. reflexivity.
  - rewrite IHreturn, IHexn. reflexivity.
Qed.

Theorem rename_permute_state : rename_permute_state_statement.
Proof.
  intros r p [e vars conts statics hp rd].
  unfold rename_state, permute_state. simpl. f_equal.
  - apply rename_permute_env.
  - unfold rename_cont_env, permute_cont_env. rewrite !map_map.
    apply map_ext. intros [k c]. simpl. rewrite rename_permute_cont_value.
    reflexivity.
Qed.

(* Reading through the action. *)

Theorem lookup_permute_heap : lookup_permute_heap_statement.
Proof.
  intros p h l. induction h as [ | [bound c] h IH]; simpl; [ reflexivity | ].
  rewrite permute_eqb. destruct (Pos.eqb l bound); [ reflexivity | exact IH ].
Qed.

Theorem is_bound_permute_heap : is_bound_permute_heap_statement.
Proof.
  intros p h l. unfold is_bound. rewrite lookup_permute_heap.
  destruct (lookup l h); reflexivity.
Qed.

Theorem lookup_permute_env : lookup_permute_env_statement.
Proof. intros p vars x. apply lookup_map_values. Qed.

Theorem kind_of_permute_value : kind_of_permute_value_statement.
Proof. intros p [ | | | | | | | | ]; reflexivity. Qed.
