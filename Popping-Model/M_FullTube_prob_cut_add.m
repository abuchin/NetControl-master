function A = M_FullTube_prob_cut_add(n,m,p,trials)

% CREATES the adjacency matrix of a size n by m, where n - rows, m -
% columns

% CUTS the connections to fit the probability distrubution p.
% ADD the connections to restore the impaired the probability distribution
% p - vector of the outcomes consisting of probabilites


function ind = graph_element(x,y,m)
%UNTITLED2 Summary of this function goes here
%   returns the number of an element on the graph, numbrering: 
% left -> right, up -> down
ind=x+(y-1)*m;
end

% N - size parameter, number of elements in the matrix - sqrt(elements)


A=zeros(m*n,m*n); % size of adjacency matrix



for x=1:1:m         % loop over all all graph edges, generate tube-like adjacency matrix
    for y=1:1:n
                         
    % tube border conditions    
        A(graph_element(x,1,m),graph_element(x,n,m))=1;        
        A(graph_element(x,n,m),graph_element(x,1,m))=1;                
                               
        
        if x>1                                                     % diagonal border
            A(graph_element(x,1,m),graph_element(x-1,n,m))=1;        
            A(graph_element(x-1,n,m),graph_element(x,1,m))=1;                
        end
        
        if x+1<=m                                                  % diagonal border
            A(graph_element(x,1,m),graph_element(x+1,n,m))=1;        
            A(graph_element(x+1,n,m),graph_element(x,1,m))=1;                
        end
        
        A(graph_element(1,y,m),graph_element(m,y,m))=0;                        
        A(graph_element(m,y,m),graph_element(1,y,m))=0;        
                    
    % formula for the rest of the elements
        
       if x>1
       A(graph_element(x,y,m),graph_element(x-1,y,m))=1;            
       A(graph_element(x-1,y,m),graph_element(x,y,m))=1;
        if y>1                                                      % diagonal elements
            A(graph_element(x,y,m),graph_element(x-1,y-1,m))=1;              
            A(graph_element(x-1,y-1,m),graph_element(x,y,m))=1;
        end
        if y+1<=n                                                   % diagonal elements
            A(graph_element(x,y,m),graph_element(x-1,y+1,m))=1;            
            A(graph_element(x-1,y+1,m),graph_element(x,y,m))=1;
        end
       end       
   
       
       if y>1      
        A(graph_element(x,y,m),graph_element(x,y-1,m))=1;
        A(graph_element(x,y-1,m),graph_element(x,y,m))=1;        
       end                                
           
       
       if x+1<=m
       A(graph_element(x,y,m),graph_element(x+1,y,m))=1;                   
       A(graph_element(x+1,y,m),graph_element(x,y,m))=1;
           if y>1                                                   % diagonal elements
            A(graph_element(x,y,m),graph_element(x+1,y-1,m))=1;            
            A(graph_element(x+1,y-1,m),graph_element(x,y,m))=1;
           end                                                     
            if y+1<=n                                               % diagonal elements
            A(graph_element(x,y,m),graph_element(x+1,y+1,m))=1;            
            A(graph_element(x+1,y+1,m),graph_element(x,y,m))=1;
            end
       end
      
       if y+1<=n
       A(graph_element(x,y,m),graph_element(x,y+1,m))=1;
       A(graph_element(x,y+1,m),graph_element(x,y,m))=1;              
       end
                            
       
    end
end


% counter
q=0;


% Randomly cutting connections

for k=1:1:100           % amount of trials to find full connected graph

p_connect = makedist('Multinomial','Probabilities',p);
B=A;                        % replace the connectivity


p_rand=random(p_connect,1,m*n); % create vector of connection number per element
     
for k=1:1:trials

% Remove connections

for i=1:1:m*n                   % loop over all all element, half of the symmetric matrix
    
    connected=find(B(i,:)>0);   % indexes of connected elements        
    % remove connections
    if length(connected) >= p_rand(i) % cut elements if there too many connections        
    connection_cut=datasample(connected,abs(length(connected)-p_rand(i)),'Replace',false);         
    B(i,connection_cut)=0;
    B(connection_cut,i)=0;
    end
    
end
 
% Add missing connections


% add more elements if too many are cut    
for i=1:1:m*n                   % loop over all the elements, half of the symmetric matrix
    
    connected=find(B(i,:)>0);   % indexes of connected elements
        
    if length(connected) < p_rand(i) % add elements if there are not enough connections
    connection_possible=find(A(i,:)>0);    
    connection_add=datasample(connection_possible,abs(length(connected)-p_rand(i)),'Replace',false);
    B(i,connection_add)=1;
    B(connection_add,i)=1;
    end
    
end

for i=1:1:m*n                   % loop over all all element, half of the symmetric matrix
    
    connected=find(B(i,:)>0);   % indexes of connected elements        
    % remove connections
    if length(connected) >= p_rand(i) % cut elements if there too many connections        
    connection_cut=datasample(connected,abs(length(connected)-p_rand(i)),'Replace',false);         
    B(i,connection_cut)=0;
    B(connection_cut,i)=0;
    end
    
end


end

%}

B_graph=graph(B);       % create the graph out of the adjacency matrix


if max(conncomp(B_graph))==1    
    break              % break the loop if all elements are parts of the same graph
else
    disp('Did not find full connected graph yet');
end


end

A=B;

end
